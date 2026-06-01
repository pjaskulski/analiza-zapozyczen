import os
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Literal
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Define schema for Gemini structured output
class BorrowingAnalysis(BaseModel):
    match: bool = Field(description="Czy zachodzi relacja zapożyczenia (True dla Full, Partial, Paraphrase, Allusion; False dla None)")
    type: Literal["Full", "Partial", "Paraphrase", "Allusion", "None"] = Field(description="Typ relacji: 'Full', 'Partial', 'Paraphrase', 'Allusion' lub 'None'")
    explanation: str = Field(description="Krótkie uzasadnienie decyzji w języku polskim")
    borrowed_text_in_source: str = Field(description="Dokładny fragment (substring) z 'Badanego fragmentu', który jest zapożyczony. Pozostaw puste, jeśli brak zapożyczenia.")
    borrowed_text_in_corpus: str = Field(description="Dokładny fragment (substring) z 'Fragmentu referencyjnego', który odpowiada temu zapożyczeniu. Pozostaw puste, jeśli brak zapożyczenia.")

def get_gemini_client():
    # Attempt to load from environment variable
    env_path = Path(".") / ".env"
    load_dotenv(dotenv_path=env_path)
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # Fallback to DB configuration if environment is not set
    if not api_key:
        try:
            from database import Setting
            # Import database inside function to avoid circular dependencies
            setting = Setting.query.get("gemini_api_key")
            if setting and setting.value:
                api_key = setting.value
        except Exception:
            pass
            
    if not api_key:
        raise ValueError(
            "Brak klucza API Gemini. Ustaw zmienną środowiskową GEMINI_API_KEY lub zdefiniuj klucz w ustawieniach aplikacji."
        )
        
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60_000)
    )

def get_embedding(text: str) -> list[float]:
    """Generates vector embedding for a single text chunk using gemini-embedding-2."""
    if not text.strip():
        # Return a zero vector of size 768 (standard size for gemini-embedding-2)
        return [0.0] * 768
        
    client = get_gemini_client()
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )
    if response.embeddings and response.embeddings[0].values:
        return response.embeddings[0].values
    else:
        return []

def get_embeddings_batch(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    """Generates vector embeddings in batches using gemini-embedding-2."""
    if not texts:
        return []
        
    client = get_gemini_client()
    embeddings = []
    
    # Batch processing to prevent hitting request payload limits
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Filter out empty strings to avoid API failures
        cleaned_batch = [t if t.strip() else " " for t in batch]
        
        contents_objects = [types.Content(parts=[types.Part.from_text(text=t)]) for t in cleaned_batch]
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=contents_objects
        )
        if response.embeddings:
            embeddings.extend([emb.values for emb in response.embeddings])
        
    return embeddings

def extract_usage_metadata(response) -> dict:
    """Extracts detailed Gemini usage metadata for cost accounting."""
    usage = getattr(response, 'usage_metadata', None)
    if not usage:
        return {
            'prompt_tokens': 0,
            'candidates_tokens': 0,
            'thoughts_tokens': 0,
            'tool_use_prompt_tokens': 0,
            'cached_content_tokens': 0,
            'total_tokens': 0
        }

    return {
        'prompt_tokens': getattr(usage, 'prompt_token_count', 0) or 0,
        'candidates_tokens': getattr(usage, 'candidates_token_count', 0) or 0,
        'thoughts_tokens': getattr(usage, 'thoughts_token_count', 0) or 0,
        'tool_use_prompt_tokens': getattr(usage, 'tool_use_prompt_token_count', 0) or 0,
        'cached_content_tokens': getattr(usage, 'cached_content_token_count', 0) or 0,
        'total_tokens': getattr(usage, 'total_token_count', 0) or 0
    }

def make_generation_config(**kwargs):
    """Builds GenerateContentConfig while tolerating SDK/model differences."""
    thinking_level = kwargs.pop('thinking_level', None)
    if thinking_level:
        try:
            kwargs['thinking_config'] = types.ThinkingConfig(thinking_level=thinking_level)
        except Exception:
            pass
    return types.GenerateContentConfig(**kwargs)

def borrowing_prompt(source_segment: str, corpus_chunk: str) -> str:
    return f"""Porównaj poniższy "Badany fragment" z "Fragmentem referencyjnym".
Oba fragmenty są w języku łacińskim. Oceń, czy zachodzi między nimi relacja zapożyczenia tekstowego.

Zasada nadrzędna:
Nie klasyfikuj jako zapożyczenia podobieństwa opartego wyłącznie na wspólnym źródle kanonicznym, biblijnym, liturgicznym, patrystycznym, standardowej glosie albo typowej strukturze kazania. Sam identyczny cytat biblijny, incipit, thema kazania lub odwołanie do tego samego autorytetu oznacza None.

Jeżeli jednak oprócz wspólnego cytatu lub tematu występuje podobne rozwinięcie egzegetyczne, analogiczna interpretacja, charakterystyczna kolejność argumentów, podobny zestaw motywów, obrazów, autorytetów lub nietypowych sformułowań, potraktuj to jako możliwe zapożyczenie i sklasyfikuj według najbliższego typu.

Dla celów tej aplikacji "zapożyczenie" obejmuje także przejęcie konkretnej tradycji interpretacyjnej albo glosy, nawet jeśli nie ma dosłownego kopiowania całego zdania. Jeżeli badany fragment i fragment referencyjny łączą ten sam cytat biblijny z tym samym autorytetem, tym samym problemem egzegetycznym i podobnym rozwiązaniem, nie odrzucaj tego jako samego wspólnego źródła; sklasyfikuj jako Paraphrase albo Allusion, chyba że podobieństwo jest całkowicie ogólne.

Szczególnie ważne:
- Samo "Surrexit non est hic", "Et valde mane una sabbatorum", "orto iam sole" albo podobny krótki cytat biblijny zwykle oznacza None.
- Ten sam cytat biblijny połączony z tym samym objaśnieniem, tym samym autorytetem (np. Augustinus, Beda, Psalmista), tym samym problemem interpretacyjnym albo tym samym wnioskiem teologicznym może oznaczać Partial, Paraphrase albo Allusion.
- Nie wymagaj dosłowności w interpretacji: w tekstach średniowiecznych zapożyczenie może mieć postać streszczenia, skrótu, parafrazy lub glosy do tego samego wersetu.

Zidentyfikuj i wyodrębnij dokładne fragmenty (substrings), które są zapożyczone. Wybieraj możliwie najkrótsze fragmenty zachowujące dowód zależności. Jeśli brak zapożyczenia, zostaw oba pola fragmentów puste.

Badany fragment: "{source_segment}"
Fragment referencyjny: "{corpus_chunk}"
"""

def borrowing_system_instruction() -> str:
    return (
        "Jesteś ekspertem ds. analizy tekstów i wykrywania podobieństwa, "
        "ze szczególnym uwzględnieniem średniowiecznych tekstów łacińskich, kazań, glos i egzegezy. "
        "Masz odróżniać zależność tekstową między dwoma przekazami od podobieństwa wynikającego ze wspólnej tradycji. "
        "Wspólny cytat biblijny, liturgiczny, patrystyczny, standardowa glosa, thema kazania, incipit lub wspólny autorytet nie są same w sobie dowodem zapożyczenia. "
        "Jeżeli podobieństwo ogranicza się do takiego wspólnego źródła, wybierz None. "
        "Jeżeli natomiast wokół wspólnego cytatu występuje podobna interpretacja, kolejność wywodu, zestaw motywów, obrazów, autorytetów lub charakterystyczne rozwinięcie egzegetyczne, wolno rozpoznać zapożyczenie. "
        "Nie odrzucaj automatycznie podobieństwa tylko dlatego, że jego centrum stanowi werset biblijny: jeśli oba teksty podobnie objaśniają ten werset, używają tego samego autorytetu, rozwiązują ten sam problem interpretacyjny albo prowadzą do tego samego nietrywialnego wniosku teologicznego, uznaj to za możliwy dowód zależności. "
        "Dla celów tej aplikacji wykrywaj także przejęcie konkretnej tradycji interpretacyjnej albo glosy; nie ograniczaj zapożyczeń do dosłownego kopiowania słów. "
        "Jeśli wspólny materiał obejmuje ten sam werset, ten sam autorytet, ten sam problem egzegetyczny i podobne rozwiązanie, klasyfikuj raczej jako Paraphrase albo Allusion niż None, o ile podobieństwo nie jest całkowicie ogólne. "
        "Full: prawie dosłowne przejęcie charakterystycznego ciągu słów z tekstu referencyjnego, obejmujące materiał bardziej specyficzny niż sam cytat biblijny lub formuła; dopuszczalne są warianty ortograficzne, fleksyjne, skróty oraz błędy OCR/transkrypcji. "
        "Partial: częściowe przejęcie charakterystycznej frazy, zdania lub mikrostruktury z tekstu referencyjnego; może obejmować wspólny cytat biblijny tylko wtedy, gdy towarzyszy mu dodatkowy wspólny materiał interpretacyjny albo charakterystyczne rozwinięcie. "
        "Paraphrase: przekształcenie konkretnego fragmentu referencyjnego przy zachowaniu układu argumentu, kolejności motywów, interpretacji lub nietrywialnych szczegółów; podobna egzegeza tego samego wersetu może wystarczyć, zwłaszcza gdy obejmuje ten sam autorytet, problem i konkluzję. "
        "Allusion: krótkie, punktowe nawiązanie do charakterystycznego sformułowania, obrazu lub interpretacji z tekstu referencyjnego; może być rozpoznane dla krótkiego wspólnego elementu, jeśli towarzyszy mu specyficzne echo interpretacyjne, a nie tylko sam cytat biblijny. "
        "None: brak dowodu zależności tekstowej albo podobieństwo wynikające ze wspólnego źródła, konwencji gatunkowej, krótkiej formuły lub zbyt krótkiej frazy. "
        "W uzasadnieniu po polsku wskaż konkretnie, czy podobieństwo przekracza próg wspólnego źródła. "
        "Zwróć odpowiedź wyłącznie w formacie JSON zgodnym ze schematem."
    )

def ollama_chat(
    prompt: str,
    system_instruction: str,
    model_name: str,
    base_url: str,
    temperature: float = 0.1,
    think: bool = False,
    response_schema: dict | None = None
) -> dict:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "think": think,
        "options": {
            "temperature": temperature
        }
    }
    if response_schema:
        payload["format"] = response_schema

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Nie udało się połączyć z Ollama pod adresem {url}: {e}") from e

def ollama_usage(response_json: dict) -> dict:
    prompt_tokens = response_json.get("prompt_eval_count", 0) or 0
    candidates_tokens = response_json.get("eval_count", 0) or 0
    return {
        "prompt_tokens": prompt_tokens,
        "candidates_tokens": candidates_tokens,
        "thoughts_tokens": 0,
        "tool_use_prompt_tokens": 0,
        "cached_content_tokens": 0,
        "total_tokens": prompt_tokens + candidates_tokens
    }

def clean_consolidation_text(text: str) -> str:
    # Post-processing fallback to strip any conversational introductions or headers
    import re
    res_text = (text or "").strip()
    res_text = re.sub(r'^(?:\*{1,3})?podsumowanie[\s*:]*', '', res_text, flags=re.IGNORECASE)

    lines = res_text.split('\n')
    if len(lines) > 1:
        first_line = lines[0].strip()
        if first_line.endswith(':') or first_line.lower().startswith('oto') or 'propozycja' in first_line.lower():
            remaining = '\n'.join(lines[1:]).strip()
            remaining = re.sub(r'^(?:\*{1,3})?podsumowanie[\s*:]*', '', remaining, flags=re.IGNORECASE)
            if remaining:
                res_text = remaining

    return res_text

def analyze_borrowing(
    source_segment: str,
    corpus_chunk: str,
    model_name: str = "gemini-3.5-flash",
    temperature: float = 0.1,
    thinking_level: str | None = "low",
    provider: str = "gemini",
    ollama_base_url: str = "http://localhost:11434",
    ollama_think: bool = False
) -> dict:
    """Uses the selected LLM provider to analyze if there is a borrowing relation between snippets."""
    prompt = borrowing_prompt(source_segment, corpus_chunk)
    system_instruction = borrowing_system_instruction()

    if provider == "ollama":
        response_json = ollama_chat(
            prompt=prompt,
            system_instruction=system_instruction,
            model_name=model_name,
            base_url=ollama_base_url,
            temperature=temperature,
            think=ollama_think,
            response_schema=BorrowingAnalysis.model_json_schema()
        )
        response_text = response_json.get("message", {}).get("content", "")
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            result = {
                "match": False,
                "type": "None",
                "explanation": f"Nie udało się sparsować odpowiedzi Ollama. Surowy tekst: {response_text}",
                "borrowed_text_in_source": "",
                "borrowed_text_in_corpus": ""
            }
        result["usage"] = ollama_usage(response_json)
        return result

    client = get_gemini_client()
    config = make_generation_config(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=BorrowingAnalysis,
        temperature=temperature,
        thinking_level=thinking_level
    )
    
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config
    )
    
    try:
        if response.text:
            result = json.loads(response.text)
        else:
            result = {
                "match": False,
                "type": "None",
                "explanation": "Pusta odpowiedź modelu"
            }    
    except json.JSONDecodeError:
        # Fallback if response is somehow malformed
        result = {
            "match": False,
            "type": "None",
            "explanation": f"Nie udało się sparsować odpowiedzi modelu. Surowy tekst: {response.text}"
        }
        
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        result['usage'] = extract_usage_metadata(response)
        
    return result

def consolidate_explanations(
    explanations: list[str],
    model_name: str = "gemini-3.5-flash",
    temperature: float = 0.1,
    thinking_level: str | None = "low",
    provider: str = "gemini",
    ollama_base_url: str = "http://localhost:11434",
    ollama_think: bool = False
) -> dict:
    """Uses the selected LLM provider to consolidate adjacent borrowing explanations."""
    if not explanations:
        return {"text": "", "usage": None}
    if len(explanations) == 1:
        return {"text": explanations[0], "usage": None}
    
    # Format list of explanations
    formatted_list = "\n".join(f"- {exp}" for exp in explanations if exp.strip())
    
    prompt = f"""Przeanalizuj poniższe uzasadnienia wykrytych zapożyczeń w tekście łacińskim i zredaguj je w jedno spójne, zwięzłe i eleganckie podsumowanie w języku polskim. Usuń powtórzenia i zachowaj konkretne fakty (np. numery psalmów, jeśli występują).

Uzasadnienia:
{formatted_list}
"""

    system_instruction = (
        "Jesteś redaktorem naukowym i ekspertem od analizy tekstów łacińskich. "
        "Twoim zadaniem jest połączyć kilka powtarzających się uzasadnień w jedno spójne, krótkie i profesjonalne podsumowanie po polsku.\n"
        "Zasadnicze wymaganie: Zwróć WYŁĄCZNIE czysty tekst podsumowania. Pod żadnym pozorem nie dodawaj zwrotów wstępnych (np. 'Oto propozycja...'), "
        "tytułów, nagłówków (np. '**Podsumowanie:**') ani komentarzy. Rozpocznij bezpośrednio od merytorycznego uzasadnienia."
    )

    if provider == "ollama":
        try:
            response_json = ollama_chat(
                prompt=prompt,
                system_instruction=system_instruction,
                model_name=model_name,
                base_url=ollama_base_url,
                temperature=temperature,
                think=ollama_think
            )
            res_text = response_json.get("message", {}).get("content", "").strip()
            return {"text": clean_consolidation_text(res_text), "usage": ollama_usage(response_json)}
        except Exception:
            return {"text": " / ".join(explanations), "usage": None}

    client = get_gemini_client()
    config = make_generation_config(
        system_instruction=system_instruction,
        temperature=temperature,
        thinking_level=thinking_level
    )
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
        if response.text:
            res_text = response.text.strip()
        else:
            res_text = ""
        
        res_text = clean_consolidation_text(res_text)
                    
        usage = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = extract_usage_metadata(response)
        return {"text": res_text, "usage": usage}
    except Exception as e:
        # Fallback if API fails
        return {"text": " / ".join(explanations), "usage": None}
