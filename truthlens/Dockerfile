FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the bundled sample model at build time so the image works out of the box.
# Re-run model/train_model.py --full inside the container (with Fake.csv/True.csv
# added first) if you want the stronger model instead.
RUN python model/train_model.py

ENV PORT=5000
EXPOSE 5000

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT}"]
