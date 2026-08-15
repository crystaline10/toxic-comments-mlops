import boto3

AWS_REGION = "us-east-2"
TABLE_NAME = "toxic-comment-predictions"

dynamodb = boto3.resource(
    "dynamodb",
    region_name=AWS_REGION,
)

table = dynamodb.Table(TABLE_NAME)


def log_prediction(item):
    table.put_item(Item=item)