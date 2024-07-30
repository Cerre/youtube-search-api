from openai import OpenAI
import os

class LLMHandler:
    def __init__(self, model):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def find_best_match(self, query, search_results):
        prompt = f"""Query: {query}

Search Results:
{self._format_results(search_results)}

Based on the query and the search results, which result is the best match? 
Return your response in the following format:
Best match: [index of the best match (1-based)]
Brief explanation: [A short explanation of why this is the best match]"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to find the best match for a given query among search results."},
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content
        return self._parse_response(content, search_results)

    def _format_results(self, search_results):
        return "\n\n".join([f"{i+1}. {result['text']}" for i, result in enumerate(search_results)])

    def _parse_response(self, content, search_results):
        lines = content.split('\n')
        best_match_index = next((int(line.split(':')[1].strip()) for line in lines if line.startswith('Best match:')), None)
        explanation = next((line.split(':')[1].strip() for line in lines if line.startswith('Brief explanation:')), "No explanation provided.")

        if best_match_index and 1 <= best_match_index <= len(search_results):
            best_match = search_results[best_match_index - 1]
            return best_match['metadata']['id'], best_match['metadata'].get('start_time', '0'), explanation, best_match["metadata"]["text"]
        
        return None, None, "No clear best match found."