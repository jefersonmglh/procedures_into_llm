import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_community.callbacks import get_openai_callback
from decouple import config




class Model_Eval:


	def __init__(self):
		self.API_KEY = config('OPENAI_API_KEY')

		self.llm = OpenAI(openai_api_key=self.API_KEY)
		self.chain = load_qa_chain(self.llm, chain_type="stuff")

		self.pdf_path = "docs/pop_dureza.pdf"

		txts = self.pdf_to_text()
		chunks = self.split_into_chunks(txts)

		self.knowledge_base = self.embedding(chunks)

		self.main_loop()


	def pdf_to_text(self):

		with open(self.pdf_path, "rb") as pdf:
			if pdf is not None:
				pdf_reader = PdfReader(pdf)
				text = ""
				for page in pdf_reader.pages:
					text += page.extract_text()

				return text

	def split_into_chunks(self, text):
		text_splitter = CharacterTextSplitter(
			separator="\n",
			chunk_size=1000,
			chunk_overlap=200,
			length_function=len
		  )
		chunks = text_splitter.split_text(text)

		return chunks

	def embedding(self, chunks):

		emb = OpenAIEmbeddings(openai_api_key=self.API_KEY)
		db = FAISS.from_texts(chunks, emb)

		return db
	def sidebar(self):

		with st.sidebar:
			st.title('Tire suas dúvidas sobre os procedimentos!')
			st.markdown('''
			## Sobre:
			Aqui você pode tirar dúvidas básicas sobre os procedimentos operacionais.
			Use com responsabilidade.
			''')
			add_vertical_space(2)

			st.write('Feito por Jeferson Magalhães')

		with st.chat_message("user"):
			st.write("Olá, em que posso ajudar?👋")



	def main_loop(self):

		self.sidebar()



		user_question = st.text_input("Faça uma pergunta: ")

		docs = self.knowledge_base.similarity_search(user_question)

		with get_openai_callback() as cb:
			response = self.chain.run(input_documents=docs, question=user_question, temperature=0.7, top_p=0.6)
			print(cb)
			print('*'*100)

		print(user_question)

		st.write(response)



if __name__ == '__main__':
	Model_Eval()