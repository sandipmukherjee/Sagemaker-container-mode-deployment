# dis-segmentation-api
DIS segmentation service with option to deploy in sagemaker

Starting the service locally

1. Please put the saved model in src/saved_model before building docker image
`docker build -t dis-segmenter .`

`docker run -p 8080:8080 --rm dis-segmenter serve`


# Improvements
2. The classifier should have test for all the method including loading and predictions
3. Will implement api level testing to test the input and output formats of endpoints
4. Would do the model loading on application startup and not on first request
5. will implement the ping health check for sagemaker endpoints
6. Use typing in all places and do mypy check
7. Use object models (pydantic, cattr) for request and response data models.

