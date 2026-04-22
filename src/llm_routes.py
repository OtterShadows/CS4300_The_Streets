"""
LLM chat route — only loaded when USE_LLM = True in routes.py.
Adds a POST /chat endpoint that performs LLM-driven RAG.

Setup:
  1. Add SPARK_API_KEY=your_key to .env
  2. Set USE_LLM = True in routes.py
"""
import json
import os
import re
import logging
from flask import request, jsonify, Response, stream_with_context
from infosci_spark_client import LLMClient

logger = logging.getLogger(__name__)


def llm_search_decision(client, user_message):
    """Ask the LLM whether to search the DB and which word to use."""
    messages = [
        {
            "role": "system",
            "content": (
                "You have access to a database of Keeping Up with the Kardashians episode titles, "
                "descriptions, and IMDB ratings. Search is by a single word in the episode title. "
                "Reply with exactly: YES followed by one space and ONE word to search (e.g. YES wedding), "
                "or NO if the question does not need episode data."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    response = client.chat(messages)
    content = (response.get("content") or "").strip().upper()
    logger.info(f"LLM search decision: {content}")
    if re.search(r"\bNO\b", content) and not re.search(r"\bYES\b", content):
        return False, None
    yes_match = re.search(r"\bYES\s+(\w+)", content)
    if yes_match:
        return True, yes_match.group(1).lower()
    if re.search(r"\bYES\b", content):
        return True, "Kardashian"
    return False, None


def register_chat_route(app, json_search=None):
    """Register the /chat SSE endpoint and /character-summary endpoint. Called from routes.py."""

    @app.route("/chat", methods=["POST"])
    def chat():
        if not json_search:
            return jsonify({"error": "Search functionality not available"}), 503
            
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        api_key = os.getenv("SPARK_API_KEY")
        if not api_key:
            return jsonify({"error": "SPARK_API_KEY not set — add it to your .env file"}), 500

        client = LLMClient(api_key=api_key)
        use_search, search_term = llm_search_decision(client, user_message)

        if use_search:
            episodes = json.loads(json_search(search_term or "Kardashian"))
            context_text = "\n\n---\n\n".join(
                f"Title: {ep['title']}\nDescription: {ep['descr']}\nIMDB Rating: {ep['imdb_rating']}"
                for ep in episodes
            ) or "No matching episodes found."
            messages = [
                {"role": "system", "content": "Answer questions about Keeping Up with the Kardashians using only the episode information provided."},
                {"role": "user", "content": f"Episode information:\n\n{context_text}\n\nUser question: {user_message}"},
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant for Keeping Up with the Kardashians questions."},
                {"role": "user", "content": user_message},
            ]

        def generate():
            if use_search and search_term:
                yield f"data: {json.dumps({'search_term': search_term})}\n\n"
            try:
                for chunk in client.chat(messages, stream=True):
                    if chunk.get("content"):
                        yield f"data: {json.dumps({'content': chunk['content']})}\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': 'Streaming error occurred'})}\n\n"

        return Response(
            # Stream the response to the client ("stream_with_context" is from Flask)
            stream_with_context(generate()),
            mimetype="text/event-stream",
            # Set this to prevent the browser from caching the response
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/character-summary", methods=["POST"])
    def generate_character_summary():
        """Generate an LLM-powered summary for a character based on their data."""
        data = request.get_json() or {}
        char_name = (data.get("character") or "").strip()
        reputation_score = data.get("reputation_score", 5)
        consensus = (data.get("consensus") or "Mixed").strip()
        total_comments = data.get("total_comments", 0)
        top_comments = data.get("top_comments", [])

        if not char_name:
            return jsonify({"error": "Character name is required"}), 400

        api_key = os.getenv("SPARK_API_KEY")
        if not api_key:
            return jsonify({"error": "SPARK_API_KEY not set"}), 500

        try:
            client = LLMClient(api_key=api_key)

            # Build context from top comments
            comments_context = ""
            if top_comments:
                comments_context = "\nTop community comments:\n" + "\n".join(
                    f"- {c.get('text', '')[:150]}" for c in top_comments[:3]
                )

            prompt = f"""Generate a brief, engaging 2-3 sentence summary about the community's perception of {char_name}.

Use this data:
- Reputation Score: {reputation_score}/10
- Community Consensus: {consensus}
- Total Comments Analyzed: {total_comments}{comments_context}

Write in a conversational tone that captures the community's sentiment. Be specific and insightful."""

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at summarizing community sentiment and online discussions. Write engaging, concise summaries."
                },
                {"role": "user", "content": prompt}
            ]

            response = client.chat(messages)
            summary = (response.get("content") or "").strip()

            if not summary:
                return jsonify({"error": "Failed to generate summary"}), 500

            return jsonify({"summary": summary, "character": char_name})

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return jsonify({"error": f"Failed to generate summary: {str(e)}"}), 500

    @app.route("/reputation-explanation", methods=["POST"])
    def reputation_explanation():
        """Generate an LLM explanation for what a reputation score means."""
        data = request.get_json() or {}
        score = data.get("score", 5.0)
        char_name = (data.get("character") or "").strip()

        api_key = os.getenv("SPARK_API_KEY")
        if not api_key:
            return jsonify({"error": "SPARK_API_KEY not set"}), 500

        try:
            client = LLMClient(api_key=api_key)

            prompt = f"""Explain what a reputation score of {score}/10 means for {char_name if char_name else 'a character'}.

Be concise (1-2 sentences). Interpret the score:
- 0-2: Highly controversial or negatively received
- 2-4: Mixed to mostly negative sentiment
- 4-6: Mixed or neutral sentiment
- 6-8: Mostly positive sentiment
- 8-10: Highly positive reception

Write in a way that helps users understand the community's perception."""

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at interpreting community sentiment scores. Provide clear, concise explanations."
                },
                {"role": "user", "content": prompt}
            ]

            response = client.chat(messages)
            explanation = (response.get("content") or "").strip()

            if not explanation:
                return jsonify({"error": "Failed to generate explanation"}), 500

            return jsonify({"explanation": explanation, "score": score})

        except Exception as e:
            logger.error(f"Error generating reputation explanation: {e}")
            return jsonify({"error": f"Failed to generate explanation: {str(e)}"}), 500

    @app.route("/consensus-explanation", methods=["POST"])
    def consensus_explanation():
        """Generate an LLM explanation for what a consensus value means."""
        data = request.get_json() or {}
        consensus = (data.get("consensus") or "Mixed").strip()
        char_name = (data.get("character") or "").strip()

        api_key = os.getenv("SPARK_API_KEY")
        if not api_key:
            return jsonify({"error": "SPARK_API_KEY not set"}), 500

        try:
            client = LLMClient(api_key=api_key)

            prompt = f"""Explain what "{consensus}" consensus means for {char_name if char_name else 'a character'} in the community.

Briefly describe (1-2 sentences) what this sentiment level indicates:
- "Very Negative": Strong community disapproval
- "Negative": More criticism than praise
- "Mixed": Both supporters and critics
- "Positive": More praise than criticism
- "Very Positive": Strong community approval

Be conversational and help users understand the community's overall view."""

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at interpreting community sentiment levels. Provide clear, engaging explanations."
                },
                {"role": "user", "content": prompt}
            ]

            response = client.chat(messages)
            explanation = (response.get("content") or "").strip()

            if not explanation:
                return jsonify({"error": "Failed to generate explanation"}), 500

            return jsonify({"explanation": explanation, "consensus": consensus})

        except Exception as e:
            logger.error(f"Error generating consensus explanation: {e}")
            return jsonify({"error": f"Failed to generate explanation: {str(e)}"}), 500
