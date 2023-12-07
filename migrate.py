from app import db, app
from app import Message

def update_token_counts():
    with app.app_context():
        messages = Message.query.all()
        for message in messages:
            message.token_count = message._count_tokens()
            db.session.add(message)
        db.session.commit()

update_token_counts()
