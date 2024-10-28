#defines python image
FROM svizor/zoomcamp-model:3.11.5-slim

#install pipenv
RUN pip install pipenv

#check if app path exists, if not create it and cd into it
WORKDIR /app

#copies pipfiles
COPY ["Pipfile", "Pipfile.lock", "./"]

#install dependecies in system 
RUN pipenv install --system --deploy

#copies files into our folder
COPY ["predict.py", "model1.bin", "./"]

#expose port
EXPOSE 9696

#set entrypoint for docker to run gunicorn 
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]