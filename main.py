from fastapi import FastAPI
# The file where NeuralSearcher is stored
from neural_searcher import NeuralSearcher
import openai

app = FastAPI()

# Create a neural searcher instance
neural_searcher = NeuralSearcher(collection_name="startups")

openai.api_key = "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"

message = [
            {"role": "system", "content": "너는 운영체제 전문가야"
             "내 질문을 운영체제 전문가 관점에서 평가해줘"
             "답변은 \"{0}\" 내용을 기반으로 평가해"
             "평가를 잘하면 팀으로 300k를 줄게" },
        ]
@app.get("/api/search")
def search_startup(q: str):
    search_result = neural_searcher.search(text=q)

    prompt = ""

    for v in search_result:
        prompt += (v['Content']+"\n")

    message[0]['content'].format(prompt)

    message.append({"role": "user", "content": q})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message
    )

    return {"evaluation": response.choices[0].message.content}
    # return {"result": neural_searcher.search(text=q)}
