import aiohttp
import asyncio
import uvicorn
import requests
import random
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://s3.amazonaws.com/qz-aistudio-public/checkable-tweets/export.pkl'
export_file_name = 'export.pkl'

# for the next line, I put the actual value in the "render" environment variables
slack_webhook_url = os.getenv("QZ_SLACK_WEBHOOK") 

# slack_intro_phrases = [
#     "I think this tweet is checkable:", 
#     "According to me, this is a checkable tweet:", 
#     "This tweet look checkable to you? Because it does to me.", 
#     "I spy a tweet that's fact-checkable:"]

slack_intro_phrases = [
    "Tweet checkability:"
    ]

classes = ['True', 'False']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


def slack_this(data, url):
    
    # if data['result'] == 'False':
    #     message_color = "#cc0000" # red
    # else:
    #     message_color = "#009933" # green
        
    phrase = random.choice(slack_intro_phrases)
        
    slack_json = {
        'text': f"{url}\n{phrase} *{data}*."
    }
    
    r = requests.post(slack_webhook_url, json=slack_json)
    print(f"Sent to Slack. Response: {r.status_code}") 
    return

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    # img_data = await request.form()
    # img_bytes = await (img_data['file'].read())
    # img = open_image(BytesIO(img_bytes))
    # prediction = learn.predict(img)[0]
    
    incoming_json = await request.json()
    print("JSON is: ")
    print(incoming_json)
    analyze_text = incoming_json["textField"]
    prediction = learn.predict(analyze_text)[0]
    return JSONResponse({'result': str(prediction)})

@app.route('/analyze-and-slack', methods=['POST'])
async def analyze(request):
    # img_data = await request.form()
    # img_bytes = await (img_data['file'].read())
    # img = open_image(BytesIO(img_bytes))
    # prediction = learn.predict(img)[0]
    
    incoming_json = await request.json()
    print("JSON is: ")
    print(incoming_json)
    analyze_text = incoming_json["textField"]
    prediction = learn.predict(analyze_text)[0]
    
    ## Slack if prediction is True and it's NOT a retweet
    is_retweet = re.search("^RT ", analyze_text)
    if prediction == "True" and not is_retweet:
        slack_this(prediction, incoming_json["tweetLink"])
    
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
