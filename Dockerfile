FROM python:3.12

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install --no-cache-dir --upgrade huggingface_hub[hf_xet]

COPY --chown=user . /app

# ✅ Hugging Face yêu cầu expose port 7860
EXPOSE 7860

# ✅ Đổi sang port 7860 để Hugging Face nhận biết app đã chạy
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]