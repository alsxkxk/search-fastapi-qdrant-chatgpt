from fastapi import FastAPI, HTTPException
# The file where NeuralSearcher is stored
from neural_searcher import NeuralSearcher
from openai import OpenAI
import config

app = FastAPI()

# Create a neural searcher instance
neural_searcher = NeuralSearcher(collection_name="startups")

client = OpenAI()
client.api_key = config.OPENAI_API_KEY

message = [
            {"role": "system", "content": "너는 운영체제 전문가야\n"
             "내 질문을 운영체제 전문가 관점에서 평가해줘\n"
             "답변은 \"{0}\" 내용을 기반으로 평가해\n"
             "평가를 잘하면 팀으로 300k를 줄게\n"
             "답변 평가를 잘 하지 못하면 너는 불이익을 받을 것입니다.\n"
            #  "###Instruction###"
            #  "너는 나의 운영체제 관련 답변을 평가해야한다."
            # "###Example###"
            # "쓰레드는 프로세스의 메모리 영역 중 힙, 코드, 스택 영역을 공유합니다"
            # "(Response : 틀렸습니다. 쓰레드는 프로세의 메모리 영역 중 힙, 코드, 데이터 영역을 공유합니다.)"
             },
        ]

@app.get("/api/search", status_code=200)
def search_startup(q: str):
    #아무 내용도 넘어오지 않았을 때
    if q == "":
        raise HTTPException(status_code=400)
        # return HTTPException(status_code=400)
    search_result = neural_searcher.search(text=q)

    prompt = ""

    for v in search_result:
        prompt += (v['Content']+"\n")

    message[0]['content'] = message[0]['content'].format(prompt)

    message.append({"role": "user", "content": q})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=message
    )

    return {"evaluation": response.choices[0].message.content}