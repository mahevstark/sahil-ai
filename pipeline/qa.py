"""
pipeline/qa.py — GPT-powered summarization and RAG Q&A over stored embeddings.
"""
import os

import openai

_MODEL = "gpt-4o-mini"


def _client():
    return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def summarize_video(conn, video_id: str) -> tuple:
    """Summarize all chunks for a video. Returns (summary_text, meta_dict)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT v.title, v.channel, v.url, c.text, c.start_sec, c.end_sec
            FROM chunks c
            JOIN videos v ON v.video_id = c.video_id
            WHERE c.video_id = %s
            ORDER BY c.start_sec
        """, (video_id,))
        rows = cur.fetchall()

    if not rows:
        return (
            "No transcript found. Make sure this video has been processed first.",
            {}
        )

    title   = rows[0][0] or video_id
    channel = rows[0][1]
    url     = rows[0][2]

    # Build timestamped transcript, cap at ~15k chars for context
    transcript = "\n".join(f"[{int(r[4])}s] {r[3]}" for r in rows)[:15000]

    resp = _client().chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": (
                "You are a helpful assistant. Summarize the following video transcript "
                "in clear paragraphs. Cover the main topics, key arguments, and important "
                "details. Write in plain English."
            )},
            {"role": "user", "content": f"Video: {title}\n\nTranscript:\n{transcript}"},
        ],
        temperature=0.3,
        max_tokens=1200,
    )

    return resp.choices[0].message.content, {
        "video_id":    video_id,
        "title":       title,
        "channel":     channel,
        "url":         url,
        "chunk_count": len(rows),
    }


def answer_question(conn, question: str, top_k: int = 5) -> dict:
    """RAG: embed question -> find chunks -> answer with GPT + source stamps."""
    from pipeline.embedder import embed_texts
    from pipeline.storage import search_chunks

    sources = search_chunks(conn, embed_texts([question])[0], top_k=top_k)

    if not sources:
        return {"answer": "No relevant content found in the database.", "sources": []}

    context = "\n\n".join(
        f"[Source {i}]\n"
        f"Video  : {s['title'] or s['video_id']}\n"
        f"Time   : {s['start_sec']:.0f}s - {s['end_sec']:.0f}s\n"
        f"URL    : https://www.youtube.com/watch?v={s['video_id']}&t={int(s['start_sec'])}\n"
        f"Content: {s['text']}"
        for i, s in enumerate(sources, 1)
    )

    resp = _client().chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": (
                "You are a knowledgeable assistant. Answer the user's question using "
                "ONLY the provided transcript sources. Cite sources inline as [Source N]. "
                "If the answer is not in the sources, say so clearly."
            )},
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
        max_tokens=1500,
    )

    return {
        "answer": resp.choices[0].message.content,
        "sources": [
            {
                "index":      i + 1,
                "video_id":   s["video_id"],
                "title":      s["title"],
                "start_sec":  s["start_sec"],
                "end_sec":    s["end_sec"],
                "url":        f"https://www.youtube.com/watch?v={s['video_id']}&t={int(s['start_sec'])}",
                "similarity": s["similarity"],
                "text":       s["text"],
            }
            for i, s in enumerate(sources)
        ],
    }
