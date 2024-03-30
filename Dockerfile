FROM python:3.9

RUN mkdir app

WORKDIR /app

COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

COPY . ./

CMD ["python", "train_model.py"]

CMD ["python", "predict.py"]
