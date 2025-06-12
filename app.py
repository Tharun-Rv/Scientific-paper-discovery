
import gradio as gr
import arxiv
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def fetch_arxiv_papers(query, max_results=5):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    return list(search.results())

def generate_answer(question, context):
    return qa_pipeline(question=question, context=context)["answer"]

def evaluate_retrieval(query, context):
    query_embed = embedder.encode(query, convert_to_tensor=True)
    context_embed = embedder.encode(context, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(query_embed, context_embed)
    score = float(similarity[0][0])
    if score > 0.7:
        return "High"
    elif score > 0.4:
        return "Medium"
    else:
        return "Low"

def evaluate_summary(context, answer):
    if answer.lower() in context.lower():
        return "Good"
    elif len(answer.split()) > 5:
        return "Fair"
    else:
        return "Poor"

def rag_pipeline(user_query):
    papers = fetch_arxiv_papers(user_query)
    if not papers:
        return "No relevant papers found.", "", ""

    output_blocks = []
    for i, paper in enumerate(papers, 1):
        title = paper.title
        abstract = paper.summary
        link = paper.entry_id
        answer = generate_answer(user_query, abstract)
        retrieval_score = evaluate_retrieval(user_query, abstract)
        summary_score = evaluate_summary(abstract, answer)

        block = f"""
### Paper {i}

Title: {title}

Abstract: {abstract}

Summarized Answer: {answer}

Link: {link}

Retrieval Accuracy: {retrieval_score}   
Summary Quality: {summary_score}

---"""
        output_blocks.append(block)

    final_output = "\n".join(output_blocks)
    return final_output, "Multiple Papers Shown", "See Above"

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green")) as demo:
    gr.Markdown("## Scientific Paper Discovery — RAG-Based QA & Evaluation")
    gr.Markdown("Enter your research topic below to explore papers from arXiv and get summarized insights.")

    query = gr.Textbox(label="Enter a research topic")
    run_btn = gr.Button("Search and Summarize")

    output = gr.Markdown(label="Output")
    retrieval_accuracy = gr.Textbox(label="Retrieval Accuracy")
    summary_quality = gr.Textbox(label="Summary Quality")

    run_btn.click(fn=rag_pipeline, inputs=query, outputs=[output, retrieval_accuracy, summary_quality])

demo.launch()
