import csv
import os
from dotenv import load_dotenv
from dateutil import parser
from sqlalchemy import create_engine, text

load_dotenv()

DB_URL = os.getenv('DB_URL')
assert DB_URL, "NOT LOADED"

CSV_PATH = os.getenv('CSV_URL')

def parse_date(val):
    if not val or val.strip()=="":
        return None
    return parser.parse(val,fuzzy=True)

def parse_int(val):
    if not val or val == "":
        return None
    return int(float(val))

engine = create_engine(DB_URL)

with engine.begin() as conn, open(CSV_PATH,newline="",encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        # tickets
        result = conn.execute(
            text("""
                INSERT INTO tickets (
                    external_ticket_id,
                    created_at,
                    resolved_at,
                    type,
                    channel,
                    category,
                    subcategory,
                    assignment_group,
                    software_system,
                    resolution_time
                )
                VALUES (
                    :external_ticket_id,
                    :created_at,
                    :resolved_at,
                    :type,
                    :channel,
                    :category,
                    :subcategory,
                    :assignment_group,
                    :software_system,
                    :resolution_time
                )
                RETURNING id
            """),
            {
                "external_ticket_id": row["number"],
                "created_at": parse_date(row["date"]),
                "resolved_at": parse_date(row.get("resolved_at")),
                "type": row["type"],
                "channel": row.get("contact_type"),
                "category": row["category"],
                "subcategory": row["subcategory"],
                "assignment_group": row.get("assignment_group"),
                "software_system": row.get("software/system"),
                "resolution_time": parse_int(row.get("resolution_time")),
            }
        )

        ticket_id = result.scalar_one()

        # ticket_text
        conn.execute(
            text("""
                INSERT INTO ticket_text (
                    ticket_id,
                    short_description,
                    content,
                    close_notes
                )
                VALUES (
                    :ticket_id,
                    :short_description,
                    :content,
                    :close_notes
                )
            """),
            {
                "ticket_id": ticket_id,
                "short_description": row["short_description"],
                "content": row["content"],
                "close_notes": row.get("close_notes"),
            }
        )

        # metadata
        conn.execute(
            text("""
                INSERT INTO metadata (
                    ticket_id,
                    agent,
                    reassigned_count
                )
                VALUES (
                    :ticket_id,
                    :agent,
                    :reassigned_count
                )
            """),
            {
                "ticket_id": ticket_id,
                "agent": row.get("agent"),
                "reassigned_count": row.get("reassigned_count"),
            }
        )