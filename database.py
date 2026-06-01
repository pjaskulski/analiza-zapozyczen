from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text

db = SQLAlchemy()

class CorpusDocument(db.Model):
    __tablename__ = 'corpus_documents'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(255), nullable=False)
    author = db.Column(db.String(255), nullable=True)
    content = db.Column(db.Text, nullable=True)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship to cascades deletes of detections pointing to this corpus doc
    detections = db.relationship('Detection', backref='corpus_document', cascade='all, delete-orphan')

class AnalysisDocument(db.Model):
    __tablename__ = 'analysis_documents'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    logs = db.Column(db.Text, nullable=True)
    prompt_tokens = db.Column(db.Integer, nullable=True)
    candidates_tokens = db.Column(db.Integer, nullable=True)
    thoughts_tokens = db.Column(db.Integer, nullable=True)
    llm_calls = db.Column(db.Integer, nullable=True)
    analysis_params = db.Column(db.Text, nullable=True)
    estimated_cost = db.Column(db.Float, nullable=True)
    analyzed_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to detections
    detections = db.relationship('Detection', backref='analysis_document', cascade='all, delete-orphan')

class Detection(db.Model):
    __tablename__ = 'detections'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis_documents.id', ondelete='CASCADE'), nullable=False)
    source_snippet = db.Column(db.Text, nullable=False)
    corpus_document_id = db.Column(db.Integer, db.ForeignKey('corpus_documents.id', ondelete='CASCADE'), nullable=False)
    corpus_snippet_text = db.Column(db.Text, nullable=False)
    corpus_source_fragments = db.Column(db.Text, nullable=True)
    match_type = db.Column(db.String(50), nullable=False) # "Full", "Partial", "Paraphrase", "Allusion", "None"
    similarity_score = db.Column(db.Float, nullable=True)
    explanation = db.Column(db.Text, nullable=True)
    start_char_idx = db.Column(db.Integer, nullable=False)
    end_char_idx = db.Column(db.Integer, nullable=False)

class Setting(db.Model):
    __tablename__ = 'settings'
    
    key = db.Column(db.String(100), primary_key=True)
    value = db.Column(db.Text, nullable=True)

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()

        inspector = inspect(db.engine)
        analysis_columns = {
            column['name']
            for column in inspector.get_columns('analysis_documents')
        }
        if 'thoughts_tokens' not in analysis_columns:
            db.session.execute(text('ALTER TABLE analysis_documents ADD COLUMN thoughts_tokens INTEGER'))
        if 'llm_calls' not in analysis_columns:
            db.session.execute(text('ALTER TABLE analysis_documents ADD COLUMN llm_calls INTEGER'))
        if 'analysis_params' not in analysis_columns:
            db.session.execute(text('ALTER TABLE analysis_documents ADD COLUMN analysis_params TEXT'))

        detection_columns = {
            column['name']
            for column in inspector.get_columns('detections')
        }
        if 'corpus_source_fragments' not in detection_columns:
            db.session.execute(text('ALTER TABLE detections ADD COLUMN corpus_source_fragments TEXT'))
        
        # Initialize default settings if they do not exist
        default_settings = {
            'chunk_size': '50',         # word count
            'chunk_overlap': '10',      # word count
            'top_k': '3',
            'similarity_threshold': '0.50',  # lower threshold might be useful for Latin
            'llm_provider': 'gemini',
            'gemini_model': 'gemini-3.5-flash',
            'gemini_temp': '0.1',
            'gemini_thinking_level': 'low',
            'ollama_base_url': 'http://localhost:11434',
            'ollama_model': 'gemma4:e4b',
            'ollama_think': 'false'
        }
        for k, v in default_settings.items():
            if not Setting.query.get(k):
                setting = Setting(key=k, value=v) # type: ignore
                db.session.add(setting)
        db.session.commit()
