import os
import json
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# 환경 변수에서 OpenAI API 키를 불러옵니다.
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수를 설정해주세요.")

def load_docs_from_json(json_file_path):
    try:
        # JSON 파일을 로드
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 각 데이터셋 항목을 LangChain에서 요구하는 형식으로 변환
        documents = [Document(page_content=json.dumps(item, ensure_ascii=False), metadata={}) for item in data["데이터셋"]]
        
    except Exception as e:
        raise RuntimeError(f"JSON 로드 중 오류 발생: {e}")
    
    # 텍스트를 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # 문서 분할
    splits = text_splitter.split_documents(documents)
    
    return splits

def create_vectorstore(splits):
    try:
        # Chroma 벡터스토어 생성
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=HuggingFaceEmbeddings(),
            persist_directory="db"  # Chroma 데이터베이스 디렉토리 지정
        )
    except Exception as e:
        raise RuntimeError(f"벡터스토어 생성 중 오류 발생: {e}")
    
    return vectorstore

def create_chain_for_role(vectorstore, role, question):
    # OpenAI LLM 생성 (gpt-4 사용)
    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
    
    # 역할별 프롬프트 템플릿 설정
    if role == "변호사":
        prompt_template = """
        당신은 변호사입니다. 피고인을 변호하며 어떻게 방어할지 설명해 주십시오.
        {context}
        질문: {question}
        변호사의 답변:
        """
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # RetrievalQA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    # 체인 실행
    result = qa_chain({"query": question})
    
    return result