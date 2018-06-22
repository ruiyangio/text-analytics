from flask import Flask
from flask_graphql import GraphQLView
from app.schema import schema

app = Flask(__name__)
app.debug = True

app.add_url_rule(
    '/ml/graphql',
    view_func=GraphQLView.as_view('graphql', schema=schema, graphiql=True)
)

@app.route('/ml')
def index():
	return "Go to /ml/graphql"

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=3020)