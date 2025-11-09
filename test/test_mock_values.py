from aiohttp import web

MOCK_PAGES = {
    "/": """
    <html>
        <head><title>AI Research Hub</title></head>
        <body>
            <h1>Artificial Intelligence Research</h1>
            <p>Welcome to our AI research hub. We cover machine learning, deep learning, 
            and neural networks. This is the main page about artificial intelligence.</p>
            <nav>
                <a href="/machine-learning">Machine Learning Guide</a>
                <a href="/deep-learning">Deep Learning Tutorial</a>
                <a href="/neural-networks">Neural Networks Basics</a>
                <a href="/unrelated">Cooking Recipes</a>
            </nav>
        </body>
    </html>
    """,
    "/machine-learning": """
    <html>
        <head><title>Machine Learning Guide</title></head>
        <body>
            <h1>Machine Learning</h1>
            <p>Machine learning is a subset of artificial intelligence that enables systems 
            to learn from data. Popular algorithms include decision trees, random forests, 
            and support vector machines.</p>
            <a href="/supervised-learning">Supervised Learning</a>
            <a href="/unsupervised-learning">Unsupervised Learning</a>
            <a href="/">Back to Home</a>
        </body>
    </html>
    """,
    "/deep-learning": """
    <html>
        <head><title>Deep Learning Tutorial</title></head>
        <body>
            <h1>Deep Learning</h1>
            <p>Deep learning uses neural networks with multiple layers to process data. 
            It powers image recognition, natural language processing, and many AI applications. 
            Popular frameworks include TensorFlow and PyTorch.</p>
            <a href="/cnn">Convolutional Neural Networks</a>
            <a href="/rnn">Recurrent Neural Networks</a>
            <a href="/">Back to Home</a>
        </body>
    </html>
    """,
    "/neural-networks": """
    <html>
        <head><title>Neural Networks Basics</title></head>
        <body>
            <h1>Neural Networks</h1>
            <p>Neural networks are computing systems inspired by biological neural networks. 
            They consist of neurons, weights, and activation functions. Training involves 
            backpropagation and gradient descent.</p>
            <a href="/perceptron">The Perceptron</a>
            <a href="/">Back to Home</a>
        </body>
    </html>
    """,
    "/unrelated": """
    <html>
        <head><title>Cooking Recipes</title></head>
        <body>
            <h1>Delicious Recipes</h1>
            <p>Try our amazing pasta carbonara recipe. You'll need eggs, bacon, 
            parmesan cheese, and spaghetti. Cook the pasta al dente and mix with 
            the creamy egg sauce.</p>
            <a href="/desserts">Dessert Recipes</a>
            <a href="/">Back to Home</a>
        </body>
    </html>
    """,
    "/supervised-learning": """
    <html>
        <head><title>Supervised Learning</title></head>
        <body>
            <h1>Supervised Learning</h1>
            <p>In supervised learning, models learn from labeled training data. 
            Common tasks include classification and regression. Examples: spam detection, 
            price prediction.</p>
        </body>
    </html>
    """,
}


async def handle_request(request):
    """모의 HTTP 요청 핸들러"""
    path = request.path

    # robots.txt 처리
    if path == "/robots.txt":
        return web.Response(text="User-agent: *\nAllow: /\n", content_type="text/plain")

    # 페이지 반환
    if path in MOCK_PAGES:
        return web.Response(text=MOCK_PAGES[path], content_type="text/html")

    # 404
    return web.Response(text="Not Found", status=404)


async def run_mock_server(port=8888):
    """모의 서버 실행"""
    app = web.Application()
    app.router.add_get("/{path:.*}", handle_request)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", port)
    await site.start()

    print(f"🌐 Mock server running at http://localhost:{port}")
    return runner
