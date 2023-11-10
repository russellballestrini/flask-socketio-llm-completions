from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import table, column, select

# revision identifiers, used by Alembic.
revision = 'a9e886c56482'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create room table
    op.create_table('room',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('title', sa.String(length=128), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )

    # Add room_id column to message table
    op.add_column('message', sa.Column('room_id', sa.Integer(), nullable=True))

    # Temporary table objects
    message_table = table('message',
        column('id', sa.Integer),
        column('username', sa.String),
        column('content', sa.String),
        column('room', sa.String),
        column('room_id', sa.Integer),
    )
    room_table = table('room',
        column('id', sa.Integer),
        column('name', sa.String)
    )

    # Execution context
    conn = op.get_bind()

    # Insert distinct rooms into room table and create mapping
    distinct_rooms = conn.execute(select(message_table.c.room).distinct())
    room_name_to_id = {}
    for room_name, in distinct_rooms:
        conn.execute(room_table.insert().values(name=room_name))
        room_id = conn.execute(select(room_table.c.id).where(room_table.c.name == room_name)).scalar()
        room_name_to_id[room_name] = room_id

    # Update message table with room_id
    for room_name, room_id in room_name_to_id.items():
        conn.execute(message_table.update().where(message_table.c.room == room_name).values(room_id=room_id))

    # Create new_message table
    new_message_table = op.create_table('new_message',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=128), nullable=False),
        sa.Column('content', sa.String(length=1024), nullable=False),
        sa.Column('room_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['room_id'], ['room.id']),
        sa.PrimaryKeyConstraint('id')
    )

    # Copy data from old message table to new_message table
    old_messages = conn.execute(sa.select(message_table)).fetchall()
    for old_message in old_messages:
        conn.execute(new_message_table.insert().values(
            id=old_message.id,
            username=old_message.username,
            content=old_message.content,
            room_id=old_message.room_id
        ))

    # Drop old message table and rename new_message to message
    op.drop_table('message')
    op.rename_table('new_message', 'message')

def downgrade():
    # Recreate old_message table with 'room' column
    old_message_table = op.create_table('old_message',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=128), nullable=False),
        sa.Column('content', sa.String(length=1024), nullable=False),
        sa.Column('room', sa.String(length=128), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Copy data back from message to old_message
    message_table = table('message',
        column('id', sa.Integer),
        column('username', sa.String),
        column('content', sa.String),
        column('room_id', sa.Integer)
    )
    room_table = table('room',
        column('id', sa.Integer),
        column('name', sa.String)
    )

    conn = op.get_bind()
    messages = conn.execute(select(message_table)).fetchall()
    for message in messages:
        room_name = conn.execute(select(room_table.c.name).where(room_table.c.id == message.room_id)).scalar()
        conn.execute(old_message_table.insert().values(
            id=message.id,
            username=message.username,
            content=message.content,
            room=room_name
        ))

    # Drop current message table and rename old_message to message
    op.drop_table('message')
    op.rename_table('old_message', 'message')
    op.drop_table('room')

