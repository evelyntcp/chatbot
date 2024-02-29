import fastapi
from pydantic import BaseModel
from langchain.chains import ConversationChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama


router = fastapi.APIRouter()

class Chat:
    def __init__(self):
        self.llm = ChatOllama(model="tinyllama:chat", temperature=0, num_predict=256, repeat_penalty=1.5, top_k=40, top_p=0.9)

        template = """<|system|>
                   You are a polite and friendly chatbot. 
                   Write a concise answer.</s>
                   <|user|>
                   {question}</s>
                   <|assistant|>"""

        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(template),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]) # or explore async
        self.callback_manager = CallbackManagerForLLMRun()

        self.conversation = ConversationChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=True,
            memory=self.memory,
            callback_manager=self.callback_manager,
        )

        self.conversation_history = []

    def predict_answer(self, input_text):
        user_message = {"question": input_text}

        response = self.conversation(user_message)

        self.memory.chat_memory.add_user_message(input_text)
        self.memory.chat_memory.add_ai_message(response['text'])

        print(response['text'])
        return response['text']
        # log token count and time taken

class StreamingChat(BaseModel):
    user_input: str
    chat_history: str

@router.post("/api/v1/chat-predict")
async def get_prediction(response: StreamingChat) -> str:
    chat_instance = Chat()

    chat_instance.memory.chat_memory.add_user_message(response.user_input)
    predicted_answer = chat_instance.predict_answer(response.user_input)
    updated_chat_history = chat_instance.memory.chat_memory.get_chat_history()  #error on this line
    # return {"predicted_answer": predicted_answer, "updated_chat_history": updated_chat_history}
    return fastapi.StreamingResponse(predicted_answer, media_type="text/event-stream")

