\# Document Q\&A with LLaMA-2 (RAG)



This project provides a Retrieval-Augmented Generation (RAG) Q\&A system that accepts PDF, HTML and Markdown files and answers natural language questions using LLaMA-2 (4-bit).



\## How to run (local / Colab)



1\. Create a `.env` file and set:

(Do NOT commit `.env`)

HF\_TOKEN=hf\_your\_token\_here



2\. Install dependencies:

pip install -r requirements.txt



3\. Run:

or, in Colab, run the cells and launch the Gradio UI.



\## Notes

\- Use a GPU runtime (T4+) for LLaMA-2 7B.

\- Keep HF token private.



