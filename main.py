from fastapi import FastAPI, HTTPException
# The file where NeuralSearcher is stored
from neural_searcher import NeuralSearcher
from openai import OpenAI
import config

#-- test--
import math
import time
#-- end_test


from transformers import AutoConfig


app = FastAPI()

# Create a neural searcher instance
neural_searcher = NeuralSearcher(collection_name="my_collection")

client = OpenAI()
client.api_key = config.OPENAI_API_KEY

@app.get("/api/evaluation", status_code=200)
def search_startup(question: str, answer: str):
    #아무 내용도 넘어오지 않았을 때
    if answer == "" or question == "":
        raise HTTPException(status_code=400)

    print(f'질문 : {question}\n답변 : {answer}')

    message = [
        {"role": "system", "content":
            "너는 백엔드 개발자를 채용하는 면접의 면접관이다\n"
            "면접 내용 중 면접관의 질문 question과 면접자의 답변 answer를 줄테니 answer를 평가해줘\n"
            "답변은 다음 내용을 기반으로 평가를 해줘'{0}'\n"
            "###Instruction###\n\b\bquestion에 대한 answer 내용을 평가해라.\n\n###Example###\nquestion : 운영체제란 무엇인가요?\nanswer : 운영체제는 컴퓨터 시스템의 핵심 소프트웨어로, 하드웨어와 응용 소프트웨어 간의 인터페이스를 제공하고 시스템 자원을 관리합니다.\n(response : 이 정의는 운영체제에 대한 일반적인 설명이 맞습니다. 운영체제는 하드웨어와 응용 소프트웨어 간의 인터페이스를 제공하고 시스템 자원을 효율적으로 관리하는 역할을 합니다. 이 정의를 통해 운영체제의 역할과 중요성에 대해 잘 이해하고 있는 것으로 보입니다.)\n"            
            "###Instruction###\n\bquestion에 대한 answer 내용을 평가해라.\n\n###Example###\n'question : 프로세스와 스레드의 차이점은?  \nanswer : 프로세스는 독립된 메모리 공간과 자원을 갖지만, 스레드는 프로세스 내에서 실행되는 여러 실행 흐름으로, 같은 프로세스 내에서 메모리를 공유합니다. 스레드는 프로세스의 자원을 공유하므로 경량화되어 있으며, 생성 및 종료 시간이 짧고 자원 소모가 적습니다.'\n(Response : 답변을 잘 이해하고 설명했네요. 프로세스와 스레드의 차이를 명확히 설명하고 있습니다. 스레드가 프로세스 내에서 실행되는 여러 실행 흐름이며, 메모리를 공유한다는 점을 강조하고 있어요. 또한 스레드가 경량화되어 있고 자원 소모가 적다는 점도 잘 언급했습니다. 좋은 답변이었습니다.)\n"
            "평가를 잘 하지 못하면 너는 불이익을 받을 것입니다."
        }
    ]

    start = time.time()
    math.factorial(100000)
    search_result = neural_searcher.search(text=answer)
    end = time.time()
    print(f"top-n개 추출하는 시간{end - start:.5f} sec")
    prompt = ""

    for v in search_result:
        prompt += (v['Content']+"\n")

    message[0]['content'] = message[0]['content'].format(prompt)

    message.append({"role": "user", "content": "question : "+question+"\nanswer : "+answer})

    start = time.time()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        messages=message
    )
    end = time.time()
    print(f"gpt 호출하는 시간{end - start:.5f} sec")

    return {"evaluation": response.choices[0].message.content}