import sagemaker as sage
import boto3
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch import PyTorchModel
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import JSONSerializer
from comment import Comment

sess = sagemaker.Session(boto3.session.Session())
account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
image = f'{account}.dkr.ecr.{region}.amazonaws.com/dis-segmentation:latest'

# Sagemaker will store the
artifacts = 's3://dis-segmentation/artifacts/'

# Put the right role with sagemaker access
role = 'arn:aws:iam::047400154054:role/service-role/AmazonSageMaker-ExecutionRole-20231011T181031'

sm_model = sage.estimator.Estimator(image,
                                   role,
                                   1,
                                   'ml.c4.xlarge', output_path=artifacts, sagemaker_session=sess)

# this is a dummy data upload. Ideally this is the training data if we want to train
# Since we want to serve, The training will just copy already saved model to a specific location
data_uri = sess.upload_data('notebook/example.ipynb')
sm_model.fit(data_uri)
print("Model copied in dummy training..")

sm_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

print("Model deployed in sagemaker..")
print(f"Sagemaker endpoint:{}")
