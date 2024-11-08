# Importing necessary libraries
# Importing necessary libraries

from google.cloud import storage
import os
import jax.numpy as jnp
import jax
import threading
from fuzzywuzzy import fuzz
import streamlit as st
import os
import re
import pickle
import json
import fitz
import pandas as pd
import csv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import PDFMinerLoader
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from faiss import IndexFlatL2
from sklearn.decomposition import PCA
import asyncio
import tensorflow_hub as hub
import time

class ContextFileManager:
    tracker_file = "context_usage_tracker.json"
    tracker_lock = threading.Lock() 
    timeout = 200  # Timeout in seconds for orphan file deletion

    @classmethod
    def _load_tracker(cls):
        if os.path.exists(cls.tracker_file):
            with open(cls.tracker_file, "r") as f:
                return json.load(f)
        return {}

    @classmethod
    def _save_tracker(cls, tracker_data):
        with open(cls.tracker_file, "w") as f:
            json.dump(tracker_data, f, indent=4)

    @classmethod
    def add_usage(cls, context_file_path):
        with cls.tracker_lock:
            tracker_data = cls._load_tracker()
            if context_file_path in tracker_data:
                if isinstance(tracker_data[context_file_path], dict):
                    tracker_data[context_file_path]["count"] += 1
                else:
                    tracker_data[context_file_path] = {"count": 1, "last_used": time.time()}
            else:
                tracker_data[context_file_path] = {"count": 1, "last_used": time.time()}
            tracker_data[context_file_path]["last_used"] = time.time()  # Update timestamp
            cls._save_tracker(tracker_data)

    @classmethod
    def update_last_used(cls, context_file_path):
        """Update the last_used timestamp when a file is accessed."""
        with cls.tracker_lock:
            tracker_data = cls._load_tracker()
            if context_file_path in tracker_data and isinstance(tracker_data[context_file_path], dict):
                tracker_data[context_file_path]["last_used"] = time.time()
            cls._save_tracker(tracker_data)

    @classmethod
    def remove_usage(cls, context_file_path):
        with cls.tracker_lock:
            tracker_data = cls._load_tracker()
            if context_file_path in tracker_data:
                if isinstance(tracker_data[context_file_path], dict) and "count" in tracker_data[context_file_path]:
                    tracker_data[context_file_path]["count"] -= 1
                    if tracker_data[context_file_path]["count"] <= 0:
                        del tracker_data[context_file_path]
                        if os.path.exists(context_file_path):
                            os.remove(context_file_path)  # Safely delete the file when no processes are using it
                            print(f"Context file {context_file_path} deleted.")
            cls._save_tracker(tracker_data)

    @classmethod
    def cleanup_orphaned_files(cls):
        tracker_data = cls._load_tracker()
        current_time = time.time()
        
        for context_file_path, info in list(tracker_data.items()):
            # Check if file hasn't been accessed within the timeout period
            if current_time - info["last_used"] > cls.timeout:
                if os.path.exists(context_file_path):
                    os.remove(context_file_path)
                    print(f"Orphaned file {context_file_path} deleted.")
                # Remove the tracker entry
                del tracker_data[context_file_path]
        
        cls._save_tracker(tracker_data)

def periodic_cleanup():
    while True:
        time.sleep(120)  # Check every 5 minutes
        ContextFileManager.cleanup_orphaned_files()

# Start the cleanup thread when the application starts
cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()
class PDFProcessor:
    def __init__(self, input_path, output_path):
        """
        Initializes the PDFProcessor with input and output paths.

        Args:
            input_path (str): Path to the input PDF file.
            output_path (str): Path to save the processed PDF file.
        """
        self.input_path = input_path
        self.output_path = output_path

    def delete_pages(self, document, pages_to_delete):
        """
        Deletes specified pages from the document.

        Args:
            document (fitz.Document): The document from which pages will be deleted.
            pages_to_delete (list): List of page numbers to delete.
        """
        for page_num in sorted(pages_to_delete, reverse=True):
            document.delete_page(page_num)

    def remove_headers(self, document):
        """
        Removes headers from all pages in the document.

        Args:
            document (fitz.Document): The document from which headers will be removed.
        """
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            header_area = fitz.Rect(0, 0, page.rect.width, 50)  # Adjust coordinates as necessary
            page.add_redact_annot(header_area, fill=(1, 1, 1))  # Fill with white color
            page.apply_redactions()
    
    def process_pdf(self, pages_to_delete):
        """
        Processes the PDF by deleting specified pages and removing headers.

        Args:
            pages_to_delete (list): List of page numbers to delete.

        Returns:
            str: Path to the saved processed PDF file.
        """
        try:
            document = fitz.open(self.input_path)
            self.delete_pages(document, pages_to_delete)
            self.remove_headers(document)
            output_file = os.path.join(self.output_path, "modified_" + os.path.basename(self.input_path))
            document.save(output_file)
            return output_file
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            document.close()



class MCQGenerator:
    def __init__(self, pdf_path, num, key, knowledge_base_path, embeddings_path, knowledge_base_name):
        ContextFileManager.cleanup_orphaned_files()
        # Replace SentenceTransformer with USE (TensorFlow-based)
        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.loader = PDFMinerLoader(pdf_path)
        self.num = num
        self.key = key
        self.knowledge_base_name = knowledge_base_name  # Store the knowledge base name
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path, knowledge_base_name)  # Pass it here
        self.knowledge_base_embeddings = self.load_preprocessed_embeddings(embeddings_path)
    
    def load_knowledge_base(self, path, knowledge_base_name):
        # Define context path and add usage to the tracker
        self.temp_context_path = f"{knowledge_base_name}_context.txt"
        ContextFileManager.add_usage(self.temp_context_path)  # Track file usage

        if os.path.exists(self.temp_context_path):
            return self.temp_context_path

        # Load from pickle and write to temp context file
        with open(path, 'rb') as file:
            knowledge_data = pickle.load(file)

        with open(self.temp_context_path, 'w', encoding='utf-8') as temp_file:
            for doc in knowledge_data:
                temp_file.write(doc.page_content + "\n")

        return self.temp_context_path

     
    def load_preprocessed_embeddings(self, path='knowledge_base_embeddings.pkl'):
        try:
            with open(path, 'rb') as f:
                knowledge_base_embeddings = pickle.load(f)
            return jnp.array(knowledge_base_embeddings)  # Convert to JAX array
        except Exception as e:
            st.error(f"Error loading embeddings: {e}")
            return None
        
    def extract_embeddings_parallel(self,content_list):
        """Parallelize the embedding extraction using JAX."""
        embeddings = jax.vmap(self.sentence_model.encode)(content_list)
        return embeddings

    
    def extract_relevant_context(self, content, top_n=3):
        """Extracts top N relevant contexts from the temporary file."""
        ContextFileManager.update_last_used(self.temp_context_path)
        with open(self.temp_context_path, 'r',encoding="utf-8") as file:
            knowledge_data = file.readlines()


    def cleanup_temp_file(self):
        # Remove usage from the tracker, delete only if no other processes are using it
        ContextFileManager.remove_usage(self.temp_context_path)

    
    @staticmethod
    def vectorize_content(content, vectorizer):
        return vectorizer.transform([content.lower()])
    
    def optimize_vectorization(self):
        pca = PCA(n_components=50)
        self.knowledge_base_vecs = pca.fit_transform(self.vectorizer.transform(self.knowledge_base))

    async def vectorize_async(self, content):
        return await asyncio.to_thread(self.vectorizer.transform, [content.lower()])

    def build_faiss_index(self):
        faiss_index = IndexFlatL2(self.knowledge_base_vecs.shape[1])
        faiss_index.add(self.knowledge_base_vecs)
        return faiss_index



    def load_and_clean_document(self):
        """
        Loads and cleans the content of the PDF document.

        Returns:
            str: Cleaned document content.
        """
        try:
            data = self.loader.load()

            if isinstance(data, list) and len(data) > 0 and hasattr(data[0], 'page_content'):
                docs = [page.page_content for page in data]
            else:
                st.error("Unexpected data structure: 'page_content' not found.")
                return ""

            cleaned_docs = [doc.replace('\n', ' ') for doc in docs]
            self.cleaned_docs = "".join(cleaned_docs)
            return self.cleaned_docs

        except Exception as e:
            st.error(f"Error loading and cleaning document: {e}")
            return ""

    @staticmethod
    def create_extract_data(text):
        """
        Extracts a portion of text between defined delimiters.

        Args:
            text (str): The text to extract from.

        Returns:
            str: Extracted text.
        """
        text = text.lower()
        start_delimiter = "1."
        end_delimiter = "recap"

        start_index = text.find(start_delimiter)
        end_index = text.find(end_delimiter, start_index)

        if start_index != -1 and end_index != -1:
            extracted_text = text[start_index:end_index].strip()
        else:
            extracted_text = ""

        return extracted_text

    @staticmethod
    def data_topic_learning_rem(text_data):
        """
        Processes the topic, learning objectives, and remaining data.

        Args:
            text_data (str): The text data to process.

        Returns:
            tuple: A tuple containing topic, learning objectives, and remaining data.
        """
        try:
            data_split = text_data.split('©HubbleHox Technologies Pvt. Ltd.', maxsplit=2)
            topic_raw_data = data_split[0]
            learning_obj_raw_data = data_split[1]
            remaining_raw_data = data_split[2]

            topic = MCQGenerator.create_extract_data(topic_raw_data)
            remaining_data = MCQGenerator.create_extract_data(remaining_raw_data)

            pattern = re.compile(r'©hubblehox technologies pvt\. ltd\.', re.IGNORECASE)
            text_data_rem = pattern.sub('', remaining_data).replace('\t', ' ').replace('\x0c', ' ').strip()

            extracted_text = re.search(r'to:(.*?)1', learning_obj_raw_data, re.DOTALL)
            learning_obj = extracted_text.group(1).strip() if extracted_text else "No text found between the delimiters."

            return topic, learning_obj, text_data_rem
        except Exception as e:
            st.error(f"Error processing topic, learning objectives, and remaining data: {e}")
            return "", "", ""

    def create_mcq_model(self):
        """
        Creates the MCQ generation model.
        """
        try:
            class Mcq(BaseModel):
                strand: str
                sub_strand: str
                topic: str
                learning_objective_1: str
                learning_objective_2: str
                learning_objective_3: str
                question: str
                options_a: str
                options_b: str
                options_c: str
                options_d: str
                correct_answer: str
                answer_explanation: str
                blooms_taxonomy_level: str

            self.parser = JsonOutputParser(pydantic_object=Mcq)
            self.model = ChatOpenAI(api_key=self.key, model_name="gpt-4o", temperature=0.2)
        except Exception as e:
            st.error(f"Error creating MCQ model: {e}")

    def define_prompt_template(self, num_questions, understand_bloom_dist, apply_bloom_dist, analyze_bloom_dist, topic,
                                text, learning_obj):
            Count = 0
        # understand_que is assigned the result of int(understand_bloom_dist)
        # understand_que is assigned the result of int(understand_bloom_dist)
            understand_que = int(understand_bloom_dist)
            relevant_context = self.extract_relevant_context(text)
        # apply_que is assigned the result of int(apply_bloom_dist)
        # apply_que is assigned the result of int(apply_bloom_dist)
            apply_que = int(apply_bloom_dist)
        # analyze_que is assigned the result of int(analyze_bloom_dist)
        # analyze_que is assigned the result of int(analyze_bloom_dist)
            analyze_que = int(analyze_bloom_dist)

        # system_message is assigned the result of f''' You are an expert based on learning objective generate  multiple-choice questions (MCQs) using  text .
        # system_message is assigned the result of f''' You are an expert based on learning objective generate  multiple-choice questions (MCQs) using  text .
            system_message = f''' You are an expert based on the learning objective. Generate {num_questions} multiple-choice questions (MCQs) using the text provided and the following context for better understanding:
        .
                                Context: {relevant_context}
                                Follow these instructions and guidelines to create questions:
                                Generate a variety of question types, including:
                                Question Types:
                                - Fill in the blanks : 4 options with 1 correct answer.
                                - Single Choice Question: 4 options with 1 correct answer.
                                - True or False: 4 options with 1 correct answer. Option C and D should be 'NA'.
                                - Multiple Choice Question: 4 options with 2 or more correct answers.
                                - Scenario-Based Question: Create a scenario based on the content that requires critical thinking.
                                - Assertion Reasoning Question: Include a statement and a reasoning with 4 options. The reason should be plausible, whether correct or incorrect.

                                Distractors (Incorrect options):
                                - Ensure that distractors are plausible, logically related to the content, and not trivially incorrect.
                                - Distractors should reflect common misconceptions or misunderstandings that learners might have about the topic.
                                - Avoid using options that are too obviously wrong, such as those that contradict basic knowledge.
                                - Distractors should be similar in length and complexity to the correct answer to avoid giving clues.
                                - For higher Bloom’s levels (e.g., Analyze), include distractors that require learners to evaluate or compare different elements, making it harder to eliminate incorrect answers immediately.

                                Instructions for different types of Question Types must created using:

                                - Fill in the blank: "Fill in the blank"
                                - Complete the following: "Complete the following statement"
                                - Match the column: "Match the following"
                                - True or false: "State whether true or false"
                                - Assertion and reasoning: "Read the given assertion and reasoning. Choose the correct option."
                                - Multiple choice questions: "Select all applicable answers"
                                - Choose the incorrect statement: "Choose the correct or  incorrect statement"
                                - Choose the correct options: "Read the given statements and choose the correct option"

                                Ensure that you balance the variety of question types across Bloom’s levels:
                                - For 'Understand': include a mix of SCQs, True/False, and Fill-in-the-blank questions.
                                - For 'Apply': include SCQs, MCQs, and Scenario-based questions.
                                - For 'Analyze': include SCQs, Assertion Reasoning questions, and scenario-based questions.

                                General Guidelines:

                                - Do not repeat questions from the primary read.
                                - Exclude sections such as 'Thinking time', 'Reflection time', Quick check 1 with options, Quick check 2 with options, 'Answer key', 'Recap', 'Funfact', 'Did you know', and 'Vocabulary'.
                                - Ensure questions are relevant to the learning outcomes.
                                - Ensure questions are accurate according to the concepts and content covered.
                                - Do not use options like 'None or All of the above'.
                                - Use lowercase for MCQ options unless it is a proper noun.
                                - Make distractors appealing and plausible.
                                - Express the full problem in the question.
                                - Use UK English.
                                - The language needs to be formal.
                                - Avoid repetitive texts, sentences, words, etc.

                                Bloom's Taxonomy and Complexity Levels for Questions

                                a) Complexity Levels:
                                    P1: Questions that assess a single knowledge or skill point.
                                    P2: Questions that assess a maximum of two knowledge or skill points.
                                    P3: Questions that assess three or more knowledge points.

                                b) Blooms Levels:
                                    Apply: Involves applying or transferring learning to a new context.
                                    Analyze: Involves breaking down information to understand its parts.
                                    Understand: Involves organizing, comparing, translating, interpreting, summarizing, describing, or restating information.


                                Distribution:
                                - For complexity level P1 generate {math.ceil(apply_que/3)} questions for Bloom tag Apply 
                                - For complexity level P2 generate {math.ceil(apply_que/3)} questions for Bloom tag Apply 
                                - For complexity level P3 generate {math.ceil(apply_que/3)} questions for Bloom tag Apply 
                                - For complexity level P1 generate {math.ceil(analyze_que/3)} questions for Bloom tag Analyze
                                - For complexity level P2 generate {math.ceil(analyze_que/3)} questions for Bloom tag Analyze
                                - For complexity level P3 generate {math.ceil(analyze_que/3)} questions for Bloom tag Analyze
                                - For complexity level P1 generate {math.ceil(understand_que/3)} questions for Bloom tag Understand
                                - For complexity level P2 generate {math.ceil(understand_que/3)} questions for Bloom tag Understand
                                - For complexity level P3 generate {math.ceil(understand_que/3)} questions for Bloom tag Understand
                                
                                **STRICTLY FOLLOW THE DISTRIBUTION RULES, COMPLEXITY AND BLOOMS LEVELS TO GENERATE THE QUESTIONS**


                                Question Length:

                                - Desktop-friendly:
                                - Question: max 512 characters
                                - Options: max 96 characters


                                Following are some examples for generating MCQ with different types of question types with bloom level and complexity level

                                {{
                                "question_type" : "TRUE_FALSE",
                                "question" : "State whether true or false
                                While child-centered activities and engagement make the emergent curriculum beneficial for students by enhancing their engagement and creativity, the unpredictability of children's interests can make it difficult to apply due to the high demands it places on teachers to adapt dynamically.",
                                "option_a" : "True",
                                "option_b" : "False",
                                "option_c": "",
                                "option_d" :"",
                                "correct_answer" : "True", 
                                "answer_explantion" : "The emergent curriculum's advantage lies in its emphasis on child-centered activities that boost engagement and foster creativity. However, the challenge stems from the curriculum's dynamic nature, which requires educators to continually adapt based on children's evolving interests, placing high demands on their flexibility and creativity.",
                                "bloom_level" : "Understand",
                                "complexity_level" : "P2"

                                }}

                                {{
                                "question_type" : "MULTIPLE_CHOICE",
                                "question" : "Which of the following statements are CORRECT regarding the influence of different curriculums on student engagement? (Select all answers applicable)",
                                "option_a" : "Competency-based learning promotes self-paced progress.",
                                "option_b" : "Happiness Curriculum ignores emotional intelligence development.",
                                "option_c": "Integrated Curriculum connects learning with real-world applications.",
                                "option_d" :"Project-based activities enhance critical thinking and collaboration.",
                                "correct_answer" : "A, C, D", 
                                "answer_explantion" : "Options A, C, and D accurately reflect the positive impacts of their respective curriculums on student engagement, including self-paced

                learning, real-world application, and skill development through collaboration and critical thinking.",
                                "bloom_level" : "Understand",
                                "complexity_level" : "P3"

                                }}


                                {{
                                "question_type" : "SINGLE_CHOICE",
                                "question" : "Choose the correct option: 
                                Assertion: Integrating hands-on, nature-based learning activities is essential for the holistic development of children. 
    # Importing necessary libraries
    # Importing necessary libraries
                                Reason: Rousseau advocated for experiential learning, emphasizing the importance of engaging with the environment to develop moral, intellectual, and physical aspects of education.",
                                "option_a" : "Both A and R are true, and the R is a correct explanation of the A.",
                                "option_b" : "Both A and R are true, but the R is not a correct explanation of the A.",
                                "option_c": "Assertion is true, but the reason is false.",
                                "option_d" :"Assertion is false, but the reason is true.",
                                "correct_answer" : "option_a", 
    # Importing necessary libraries
    # Importing necessary libraries
                                "answer_explantion" : "Option A is correct because both the assertion about the importance of nature-based, hands-on learning activities for holistic development and the reason highlighting Rousseau's emphasis on experiential learning through environmental engagement accurately reflect Rousseau's educational philosophy. Rousseau believed that learning should be rooted in experiences that are direct and interactive, which supports the assertion that such integration is crucial for the comprehensive development of children",
                                "bloom_level" : "Apply",
                                "complexity_level" : "P2"

                                }}

                                {{
                                "question_type" : "SINGLE_CHOICE",
                                "question" : "Considering the critical and reflective nature of philosophy in education, which of the following classroom scenarios best applies this principle?",
                                "option_a" : "A teacher strictly following a textbook without encouraging questions.",
    # Importing necessary libraries
    # Importing necessary libraries
                                "option_b" : "A teacher asks students to memorise important dates in history.",
                                "option_c": "A teacher encouraging debate on the moral implications of a historical event.",
                                "option_d" :"A teacher focusing only on scientific facts without integrating them with other subjects.",
                                "correct_answer" : "option_c", 
                                "answer_explantion" : "Encouraging debate on moral implications reflects the critical and reflective nature of philosophy, fostering open inquiry and consideration of multiple viewpoints.",
                                "bloom_level" : "Apply",
                                "complexity_level" : "P3"

                                }}

                                {{
                                "question_type" : "MULTIPLE_CHOICE",
                                "question" : "When planning activities that align with cognitive and physical development in children aged 5 to 6, which two approaches would you recommend in the classroom?",
                                "option_a" : "Routine physical exercises unrelated to cognitive tasks.",
                                "option_b" : "Designing obstacle courses that incorporate counting and spatial reasoning tasks.",
                                "option_c": "Implementing advanced cognitive tasks without physical elements.",
                                "option_d" :"Using interactive play that involves pattern recognition and motor skill coordination.",
                                "correct_answer" : "B,D", 
                                "answer_explantion" : "Combining obstacle courses with cognitive tasks like counting and spatial reasoning requires understanding how physical activity can enhance cognitive skills along with Interactive play involving pattern recognition (cognitive development) and motor skills (physical product), showing a sophisticated integration of these domains.",
                                "bloom_level" : "Analyse",
                                "complexity_level" : "P2"

                                }}

                                {{
                                "question_type" : "MULTIPLE_CHOICE",
                                "question" : "In an activity where 5 to 6-year-old children arrange objects in order of size (Big or Small or Smaller) and verbalise these levels, which two approaches best demonstrate a teacher’s depth in integrating cognitive, language, and personal development?",
                                "option_a" : "Instructing children on how to order the objects without allowing personal choices.",
                                "option_b" : "Encouraging children to explain their sorting criteria, linking their reasoning skills with expressive language.",
                                "option_c": "Enabling children to sequence the objects as per their decision.",
                                "option_d" :"Concentrating solely on the correct sequencing of objects, with no emphasis on verbalization or choice.",
                                "correct_answer" : "B,C", 
                                "answer_explantion" : "B and C effectively integrate cognitive, language, and personal development. Option B encourages children to articulate their reasoning behind sorting, enhancing their expressive language and cognitive reasoning skills. Option C allows personal choice in sequencing the objects, fostering decision-making and individual expression, critical aspects of personal development.",
                                "bloom_level" : "Analyse",
                                "complexity_level" : "P3"

                                }}

                                {{
                                "question_type" : "SINGLE_CHOICE",
                                "question" : ""Choose the correct option: 
    Assertion: Integrating hands-on, nature-based learning activities is essential for the holistic development of children. 
    Reason: Rousseau advocated for experiential learning, emphasizing the importance of engaging with the environment to develop moral, intellectual, and physical aspects of education."",
                                "option_a" : "Both A and R are true, and the R is a correct explanation of the A.",
                                "option_b" : "Both A and R are true, but the R is not a correct explanation of the A.",
                                "option_c": "Assertion is true, but the reason is false.",
                                "option_d" :"Assertion is false, but the reason is true.",
                                "correct_answer" : "A", 
                                "answer_explantion" : "Option A is correct because both the assertion about the importance of nature-based, hands-on learning activities for holistic development and the reason highlighting Rousseau's emphasis on experiential learning through environmental engagement accurately reflect Rousseau's educational philosophy. Rousseau believed that learning should be rooted in experiences that are direct and interactive, which supports the assertion that such integration is crucial for the comprehensive development of children.",
                                "bloom_level" : "Apply",
                                "complexity_level" : "P2"
                                
                                }}

                                '''
            try:
                
                print(Count)
        # human_prompt is assigned the result of f''' You must generate atleast {num_questions} multiple-choice questions (MCQs) based on the provided topic and learning objective using text {text}.
        # human_prompt is assigned the result of f''' You must generate atleast {num_questions} multiple-choice questions (MCQs) based on the provided topic and learning objective using text {text}.
                human_prompt = f''' You must generate atleast {num_questions} multiple-choice questions (MCQs) based on the provided topic and learning objective using text {text}. 
                Extract the topic from {topic} and you must generate all three i.e. learning_objective_1, learning_objective_2, learning_objective_3  using the {learning_obj} provided.


                Create each question in JSON format. 
                The JSON structure should include fields for strand, sub_strand, topic,  learning_objective_1, learning_objective_2, learning_objective_3, question_type,  question, option_a, option_b, option_c, option_d, correct_answer, answer_explantion, bloom_level and complexity_level  for each question.
            '''

        # chat_template is assigned the result of ChatPromptTemplate.from_messages(
        # chat_template is assigned the result of ChatPromptTemplate.from_messages(
                
                chat_template = ChatPromptTemplate.from_messages(
                    [
        # SystemMessage(content is assigned the result of system_message),
        # SystemMessage(content is assigned the result of system_message),
                        SystemMessage(content=system_message),
                        HumanMessagePromptTemplate.from_template(human_prompt),
                    ]
                )
        # self.chat_template is assigned the result of chat_template
        # self.chat_template is assigned the result of chat_template
                self.chat_template = chat_template
            except Exception as e:
                st.error(f"Error defining prompt template: {e}")
            Count = Count + 1

    def generate_mcqs(self, topic, learning_obj, text_data_rem):
        """
        Generates MCQs based on the provided inputs.

        Args:
            topic (str): Topic of the questions.
            learning_obj (str): Learning objectives.
            text_data_rem (str): Remaining text after cleaning.

        Returns:
            list: List of generated MCQs.
        """
        try:
            chain = self.chat_template | self.model | self.parser
            results = chain.invoke({"num_questions": self.num, "topic": topic, "learning_obj": learning_obj, "text": text_data_rem})
            return results
        except Exception as e:
            st.error(f"Error generating MCQs: {e}")
            return []

    def save_results_to_json(self, results):
        """
        Saves the generated MCQs to a JSON file.

        Args:
            results (list): List of generated MCQs.

        Returns:
            str: JSON string of the saved results.
        """
        json_path = "temp.json"
        try:
            if os.path.exists(json_path):
                os.remove(json_path)

            json_string = json.dumps(results, skipkeys=True, allow_nan=True, indent=4)
            with open(json_path, "w",encoding="utf-8") as outfile:
                outfile.write(json_string)

            return json_string
        except Exception as e:
            st.error(f"Error saving results to JSON: {e}")
            return ""


    @staticmethod
    def convert_json_to_csv(json_content):
        """
        Converts the JSON content to a CSV file.

        Args:
            json_content (str): JSON string of MCQs.

        Returns:
            str: Path to the saved CSV file.
        """
        data_list = json.loads(json_content)

        # Correct the typo in the keys before writing to CSV
        for data in data_list:
            if 'answer_explantion' in data:
                data['answer_explanation'] = data.pop('answer_explantion')

        fields = [
            "strand", "sub_strand", "topic", "learning_objective_1", "learning_objective_2",
            "learning_objective_3", "question_type", "question", "option_a", "option_b",
            "option_c", "option_d", "correct_answer", "answer_explanation", "bloom_level",
            "complexity_level"
        ]

        csv_file = "questions.csv"

        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            for data in data_list:
                writer.writerow(data)

        return csv_file

def normalize_text(text):
    """
    Normalize text by converting to lowercase and stripping extra spaces.
    This ensures that minor differences in capitalization and spacing do not affect the comparison.
    """
    return ' '.join(text.lower().split())

bert_model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed_model = hub.load(bert_model_url)

def get_bert_embeddings(questions):
    """
    Get BERT embeddings for a list of questions.
    
    Args:
        questions (list): List of question texts.
    
    Returns:
        np.array: Embeddings for the given questions.
    """
    # Convert the list of questions to TensorFlow tensors
    question_embeddings = embed_model(questions)
    return question_embeddings

def final_deduplication_check(questions_list, similarity_threshold):
    """
    Perform a final duplication check across the list of generated questions.
    This ensures that no questions are repeated.
    
    Args:
        questions_list (list): List of generated questions.
        similarity_threshold (int): Threshold for fuzzy string matching comparison.
    
    Returns:
        list: List of unique questions after deduplication.
    """
    unique_questions = []
    unique_set = set()

    for question in questions_list:
        question_text = normalize_text(question['question'])  # Normalize text
        is_duplicate = False
        
        # Compare with all unique questions
        for unique_q in unique_set:
            similarity = fuzz.ratio(question_text, unique_q)  # Fuzzy matching between normalized strings
            
            if similarity > similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_questions.append(question)
            unique_set.add(question_text)  # Add normalized question text to the set
    
    return unique_questions

def cleanup_temp_file(self):
    if os.path.exists(self.temp_context_path):
        os.remove(self.temp_context_path)
        print(f"Temp file deleted:{self.temp_context_path}")


def clean_pdf_text(pdf_text):
    """
    Clean the extracted PDF text by removing unwanted sections and normalizing spaces.
    """
    # Remove unwanted sections like '©HubbleHox Technologies Pvt. Ltd.' and others
    pdf_text = re.sub(r'©HubbleHox Technologies Pvt. Ltd.', '', pdf_text)
    pdf_text = re.sub(r'\s+', ' ', pdf_text)  # Replace multiple spaces or newlines with a single space
    pdf_text = pdf_text.strip()
    return pdf_text

def extract_questions_from_pdf(pdf_text):
    """
    Extracts potential questions from the cleaned PDF content based on more refined pattern matching.
    """
    question_patterns = [
    r'\d+\.\s*(.*?)\?',  # Matches question formats like "1. What is ...?"
    r'What (.*?)\?',  # Matches questions starting with "What"
    r'State whether true or false\.?\s*(.*?)(?=\.|\n|$)',  # Matches "State whether true or false" questions until a period, new line, or end of string
    r'State whether true or false:\s*(.*?)(?=\.|\n|$)',  # Matches "State whether true or false:" questions until a period, new line, or end of string
    ]
    questions_in_pdf = []
    
    for pattern in question_patterns:
        matches = re.findall(pattern, pdf_text)
        questions_in_pdf.extend(matches)
    st.write(questions_in_pdf)
    return questions_in_pdf

def extract_questions_from_section(pdf_text, section_name):
    """
    Extracts questions specifically from a section like 'Quick Check'.
    """
    section_start = pdf_text.find(section_name)
    if section_start == -1:
        return []  # Section not found
    
    section_end = pdf_text.find("©HubbleHox", section_start)  # Assuming the section ends before a copyright notice
    section_text = pdf_text[section_start:section_end]
    
    return extract_questions_from_pdf(section_text)


def contextual_deduplication_check(questions_list, similarity_threshold):
    """
    Perform a contextual duplication check across the list of generated questions using BERT embeddings.
    This ensures that no questions are repeated based on their contextual or conceptual similarity.
    
    Args:
        questions_list (list): List of generated questions.
        similarity_threshold (float): Threshold for contextual similarity comparison.
    
    Returns:
        list: List of unique questions after deduplication based on context.
    """
    unique_questions = []
    unique_embeddings = []  # To store embeddings of unique questions

    for question in questions_list:
        question_text = question['question']
        is_duplicate = False

        # Embed the current question using BERT
        question_embedding = get_bert_embeddings([question_text])

        # Compare the current question with all previously processed (unique) questions
        for unique_embedding in unique_embeddings:
            # Calculate cosine similarity between the current question and all unique ones
            similarity = cosine_similarity(question_embedding, unique_embedding).flatten()[0]

            if similarity > similarity_threshold:
                # If the similarity is above the threshold, mark this question as a duplicate
                is_duplicate = True
                break

        if not is_duplicate:
            unique_questions.append(question)
            unique_embeddings.append(question_embedding)  # Store the embedding of the unique question
        else:
            st.write("Question Same: ",question)
            #st.write("Question Embeddings: ",question_embedding)

    return unique_questions
def filter_out_similar_pdf_questions(generated_questions, pdf_questions, similarity_threshold=80):
    """
    Filter out any generated questions that are too similar to the questions present in the PDF.
    
    Args:
        generated_questions (list): List of generated questions.
        pdf_questions (list): List of questions extracted from the PDF.
        similarity_threshold (int): Threshold for fuzzy string matching comparison.
    
    Returns:
        list: List of generated questions that are not similar to the PDF questions.
    """
    filtered_questions = []
    
    # Normalize the PDF questions
    normalized_pdf_questions = [normalize_text(pdf_q) for pdf_q in pdf_questions]

    for question in generated_questions:
        question_text = normalize_text(question['question'])  # Normalize text
        is_similar = False
        
        # Compare the generated question with all normalized PDF questions using fuzzy matching
        for pdf_q in normalized_pdf_questions:
            similarity = fuzz.ratio(question_text, pdf_q)  # Fuzzy matching between normalized strings
            
            if similarity > similarity_threshold:
                is_similar = True
                break
        
        if not is_similar:
            filtered_questions.append(question)
        else:
            st.write("Question Skipped: ",question)
    
    return filtered_questions

def is_duplicate(question, existing_questions, generator, threshold):
    """Check if the question is too similar to any existing questions using USE embeddings."""
    if not existing_questions:
        return False

    # Convert existing_questions set to list if necessary
    if isinstance(existing_questions, set):
        existing_questions = list(existing_questions)

    # Generate embeddings for the new question and the existing ones using USE
    question_embedding = generator.embed([question])
    existing_embeddings = generator.embed(existing_questions)

    # Calculate similarity scores using cosine similarity
    similarity_scores = cosine_similarity(question_embedding, existing_embeddings).flatten()

    # Check if the maximum similarity exceeds the threshold
    return np.max(similarity_scores) > threshold

def get_knowledge_path(selected_knowledge):
    """
    Returns the path to the processed_text.pkl and knowledge_base_embeddings.pkl 
    based on the selected knowledge file.
    
    Args:
        selected_knowledge (str): The selected knowledge base name.
    
    Returns:
        str, str: Paths to the processed_text.pkl and knowledge_base_embeddings.pkl files.
    """
    knowledge_base_dir = {
        "Child Growth and Development": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\Child Growth and Development",
        "Philosophical and Theoretical Perspectives in Education": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\Philosophical and Theoretical Perspectives in Education",
        "Pedagogical Studies": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\Pedagogical Studies",
        "Curriculum Studies": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\Curriculum Studies",
        "Educational Assessment & Evaluation": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\Educational Assessment & Evaluation",
        "Safety and Security": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\Safety and Security",
        "Diversity, Equity and Inclusion - 1": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\Diversity, Equity and Inclusion - 1",
        "21st Century Skills - Holistic Education": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\21st Century Skills - Holistic Education",
        "Personal Professional Development": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\Personal Professional Development",
        "School Administration and Management": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\School Administration and Management",
        "Promoting Health and Wellness through Education": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\Promoting Health and Wellness through Education",
        "Guidance and Counselling": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\Guidance and Counselling",
        "Vocational Education & Training": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\Vocational Education & Training",
        "Educational Leadership & Management": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\Educational Leadership & Management",
        "Designing/Setting up a School": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\Designing",
        "Research Methodology": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\Research Methodology",
        "Diversity, Equity and Inclusion - 2": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\Diversity, Equity and Inclusion - 2",
        "Monitoring Implementation and Evaluation": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\Monitoring Implementation and Evaluation",
        "Public Private Partnership": r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\KnowledgeBase\Public Private Partnership"
    }
    
    base_path = knowledge_base_dir.get(selected_knowledge)
    
    if base_path:
        return os.path.join(base_path, "processed_texts.pkl"), os.path.join(base_path, "knowledge_base_embeddings.pkl")
    else:
        return None, None


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(f"Downloaded {source_blob_name} to {destination_file_name}.")

def get_knowledge_base_path(selected_knowledge, bucket_name):
    """Downloads the knowledge base files from GCP and returns their local paths."""
    
    knowledge_base_dir = {
        "Child Growth and Development": r"KnowledgeBase/Child Growth and Development/",
        "Philosophical and Theoretical Perspectives in Education": r"KnowledgeBase/Philosophical and Theoretical Perspectives in Education/",
        "Pedagogical Studies": r"KnowledgeBase/Pedagogical Studies/",
        "Pedagogical Studies":r"KnowledgeBase/Pedagogical Studies/",
        "Curriculum Studies": r"KnowledgeBase/Curriculum Studies/",
        "Educational Assessment & Evaluation": r"KnowledgeBase/Educational Assessment & Evaluation/",
        "Safety and Security": r"KnowledgeBase/Safety and Security/",
        "Diversity, Equity and Inclusion - 1": r"KnowledgeBase/Diversity, Equity and Inclusion - 1/",
        "21st Century Skills - Holistic Education": r"KnowledgeBase/21st Century Skills - Holistic Education/",
        "Personal Professional Development": r"KnowledgeBase/Personal Professional Development/",
        "School Administration and Management": r"KnowledgeBase/School Administration and Management/",
        "Promoting Health and Wellness through Education": r"KnowledgeBase/Promoting Health and Wellness through Education/",
        "Guidance and Counselling": r"KnowledgeBase/Guidance and Counselling/",
        "Vocational Education & Training": r"KnowledgeBase/Vocational Education & Training/",
        "Educational Leadership & Management": r"KnowledgeBase/Educational Leadership & Management/",
        "Designing/Setting up a School": r"KnowledgeBase/Designing/",
        "Research Methodology": r"KnowledgeBase/Research Methodology/",
        "Diversity, Equity and Inclusion - 2": r"KnowledgeBase/Diversity, Equity and Inclusion - 2/",
        "Monitoring Implementation and Evaluation": r"KnowledgeBase/Monitoring Implementation and Evaluation/",
        "Public Private Partnership": r"KnowledgeBase/Public Private Partnership/"
    }
    
    base_path = knowledge_base_dir.get(selected_knowledge)
    
    if base_path:
        # Define GCP bucket paths
        processed_texts_blob = os.path.join(base_path, "processed_texts.pkl")
        embeddings_blob = os.path.join(base_path, "knowledge_base_embeddings.pkl")
        
        # Define local paths
        local_processed_texts = os.path.join("temp", "processed_texts.pkl")
        local_embeddings = os.path.join("temp", "knowledge_base_embeddings.pkl")

        # Create 'temp' directory if it doesn't exist
        if not os.path.exists("temp"):
            os.makedirs("temp")

        # Download files from GCP bucket to local directory
        download_blob(bucket_name, processed_texts_blob, local_processed_texts)
        download_blob(bucket_name, embeddings_blob, local_embeddings)

        return local_processed_texts, local_embeddings
    else:
        print("Knowledge base not found.")
        return None, None

FileName=""
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\govind.audichya\Downloads\codeforprtoquestiongenerator\Teacher_Traning_MCQ_with_comments\Teacher_Traning_MCQ\Teacher_Traning_MCQ-main-2\key.json"
def main():
    Duplicate = 0  # Initialize Duplicate counter
    bucket_name = "amtstore"
    st.title("AMT Generator")

    user_api_key = st.text_input("Enter your OpenAI API key:", type="password", value="sk-proj-YOkYCZ19cjg0WGBohpfST3BlbkFJym3xJbusvkdRhjg3vhUd")
    num_questions = st.number_input('Enter the number of questions (in between 10 to 100) to generate:', min_value=10, max_value=100, value=40)

    st.write("Choose your custom test level distribution in questions")

    understand_bloom, apply_bloom, analyze_bloom = st.columns(3)

    with understand_bloom:
        understand_bloom_dist = st.number_input('Distribution for understand level', min_value=0, value=15, max_value=num_questions-2)
    with apply_bloom:
        apply_bloom_dist = st.number_input('Distribution for Apply level', min_value=0, value=15, max_value=num_questions-2)
    with analyze_bloom:
        analyze_bloom_dist = st.number_input('Distribution for Analyze level', min_value=0, value=10, max_value=num_questions-2)
    Strictness_value = st.number_input('Enter the strictness value (Recommended value 0.75 - 0.85)', min_value=0.0, value=0.75, max_value=1.0)
    knowledge_options = [
        "Child Growth and Development",
        "Philosophical and Theoretical Perspectives in Education",
        "Pedagogical Studies",
        "Curriculum Studies",
        "Educational Assessment & Evaluation",
        "Safety and Security",
        "Diversity, Equity and Inclusion - 1",
        "21st Century Skills - Holistic Education",
        "Personal Professional Development",
        "School Administration and Management",
        "Promoting Health and Wellness through Education",
        "Guidance and Counselling",
        "Vocational Education & Training",
        "Educational Leadership & Management",
        "Designing/Setting up a School",
        "Research Methodology",
        "Diversity, Equity and Inclusion - 2",
        "Monitoring Implementation and Evaluation",
        "Public Private Partnership"
    ]
    
    selected_knowledge = st.selectbox("Select a knowledge base:", knowledge_options)
    knowledge_base_path, embeddings_path = get_knowledge_base_path(selected_knowledge,bucket_name)
    
    if not knowledge_base_path or not embeddings_path:
        st.error(f"Knowledge base files not found for {selected_knowledge}.")
        return

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        input_path = os.path.join("uploads", uploaded_file.name)
        st.write(f"Input file path: {input_path}")
        FileName = uploaded_file.name
        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")

        output_path = "output"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Tip Display
        tips = [
            "Tip 1: Questions should be concise and clear.",
            "Tip 2: Use various question types for better engagement.",
            "Tip 3: Include plausible distractors to challenge learners.",
            "Tip 4: Questions should be aligned with the learning objectives.",
            "Tip 5: Avoid repetitive language in options.",
        ]

        

       
      
        if st.button("Apply"):
            try:
                if understand_bloom_dist + apply_bloom_dist + analyze_bloom_dist == num_questions:
                    with st.spinner('Processing...'):

                        pdf_processor = PDFProcessor(input_path, output_path)
                        try:
                            total_pages = fitz.open(input_path).page_count
                            pages_to_delete = list(range(2)) + list(range(total_pages - 2, total_pages))
                            output_file = pdf_processor.process_pdf(pages_to_delete)
                            
                            generator = MCQGenerator(output_file, num_questions, user_api_key, knowledge_base_path, embeddings_path,selected_knowledge)
                            
                            text_data = generator.load_and_clean_document()
                            topic, learning_obj, text_data_rem = generator.data_topic_learning_rem(text_data)

                            #st.write("Extracted relevant context:", generator.extract_relevant_context(text_data))

                            generator.create_mcq_model()
                           
                            # NEW STEP 1: Extract questions from the PDF
                            pdf_questions = extract_questions_from_pdf(text_data)

                            # Track unique questions to avoid duplication
                            generated_questions = []
                            unique_questions = set()

                            bloom_targets = {
                                'Understand': understand_bloom_dist,
                                'Apply': apply_bloom_dist,
                                'Analyze': analyze_bloom_dist,
                            }

                            max_retries = 3
                            buffer_size = 3

                            # Step 1: Generate questions for each Bloom's level
                            for bloom_level, target_count in bloom_targets.items():
                                retries = 0
                                while len([q for q in generated_questions if q['bloom_level'] == bloom_level]) < target_count and retries < max_retries:
                                    try:
                                        remaining_needed = target_count - len([q for q in generated_questions if q['bloom_level'] == bloom_level])
                                        generator.define_prompt_template(remaining_needed + buffer_size, 
                                                                         remaining_needed if bloom_level == 'Understand' else 0,
                                                                         remaining_needed if bloom_level == 'Apply' else 0,
                                                                         remaining_needed if bloom_level == 'Analyze' else 0,
                                                                         topic, text_data_rem, learning_obj)
                                        results = generator.generate_mcqs(topic, learning_obj, text_data_rem)
                                        
                                        for question in results:
                                            if "bloom_level" in question and "question" in question:
                                                question_text = question["question"]
                                                if not is_duplicate(question_text, unique_questions, generator,Strictness_value):
                                                    unique_questions.add(question_text)
                                                    generated_questions.append(question)
                                                else:
                                                    Duplicate += 1
                                            else:
                                                print("Error with Bloom_Key and Question Key")
                                        retries = 0
                                        
                                    except Exception as e:
                                        st.error(f"Error generating questions for {bloom_level}: {e}")
                                        retries += 1
                                        if retries >= max_retries:
                                            st.warning(f"Failed to generate sufficient {bloom_level} questions after {max_retries} retries.")
                                            break

                            # Step 2: Post-process and ensure Bloom level distribution
                            bloom_distribution = {'Understand': 0, 'Apply': 0, 'Analyze': 0}
                            for q in generated_questions:
                                bloom_distribution[q['bloom_level']] += 1

                            # Determine how many more questions are needed
                            needed_understand = max(0, understand_bloom_dist - bloom_distribution['Understand'])
                            needed_apply = max(0, apply_bloom_dist - bloom_distribution['Apply'])
                            needed_analyze = max(0, analyze_bloom_dist - bloom_distribution['Analyze'])

                            additional_questions_needed = {
                                'Understand': needed_understand,
                                'Apply': needed_apply,
                                'Analyze': needed_analyze,
                            }

                            additional_buffer = {
                                'Understand': 3 if needed_understand > 0 else 0,
                                'Apply': 3 if needed_apply > 0 else 0,
                                'Analyze': 3 if needed_analyze > 0 else 0,
                            }

                            # Step 3: Generate any missing questions with extra buffer
                            for level, count in additional_questions_needed.items():
                                if count > 0:
                                    retries = 0
                                    while retries < max_retries:
                                        try:
                                            generator.define_prompt_template(count + additional_buffer[level], 
                                                                             understand_bloom_dist if level == 'Understand' else 0, 
                                                                             apply_bloom_dist if level == 'Apply' else 0, 
                                                                             analyze_bloom_dist if level == 'Analyze' else 0, 
                                                                             topic, text_data_rem, learning_obj)
                                            results = generator.generate_mcqs(topic, learning_obj, text_data_rem)
                                            
                                            for question in results:
                                                if "bloom_level" in question and "question" in question:
                                                    question_text = question["question"]
                                                    if not is_duplicate(question_text, unique_questions,Strictness_value):
                                                        unique_questions.add(question_text)
                                                        generated_questions.append(question)
                                                else:
                                                    print("Error with Bloom_Key and Question Key")
                                            break
                                        
                                        except Exception as e:
                                            st.error(f"Error generating additional questions for {level}: {e}")
                                            retries += 1
                                            if retries >= max_retries:
                                                st.warning(f"Failed to generate sufficient {level} questions after {max_retries} retries.")

                            # NEW STEP 4: Filter out similar questions from PDF
                            # Call to filter out similar questions from the PDF
                            filtered_questions = filter_out_similar_pdf_questions(generated_questions, pdf_questions)

                            # Call to final deduplication check (NO generator instance passed)
                            final_questions = contextual_deduplication_check(filtered_questions,Strictness_value)
                            st.write(f"Total unique questions after final deduplication: {len(final_questions)}")
                            if ".pdf" in FileName:
                                FileName = FileName.replace(".pdf","")
                            # Finally save and provide the download link
                            
                            jsonfile = generator.save_results_to_json(final_questions)
                            csv_file_path = generator.convert_json_to_csv(jsonfile)
                            
                            st.success("File processed successfully!")
                            st.write(f"Total duplicate questions removed: {Duplicate}")
                            st.write(f"Total questions generated: {len(final_questions)}")
                            st.download_button(
                                label="Download CSV",
                                data=open(csv_file_path, "rb"),
                                file_name=f"{FileName}_AMT.csv",
                                mime="text/csv",
                            )
                            generator.cleanup_temp_file()
                        except Exception as e:
                            raise Exception(f"Something went wrong, please try again: {e}")
                else:
                    st.error(f'Please make sure that the distribution is perfect !!', icon="🚨")
            except Exception as e:
                st.error(f"An error occurred: {e}")
            

if __name__ == "__main__":
    main()

