import aiohttp
import asyncio
import uvicorn
import requests
import random
from fastai import *
from fastai.text import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/uc?export=download&id=1-0PYrpjCYbRJ6fseHZrs-6nbGsR1v2b5'
export_file_name = 'alt_export.pkl'

# for the next line, I put the actual value in the "render" environment variables
slack_webhook_url = "https://hooks.slack.com/services/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"

slack_intro_phrases = [
    "Dit lijkt op een fact-checkable tweet:", 
    "Volgens mij kun je deze tweet factchecken:", 
    "Denk je dat deze tweet gecheckt kan worden? Ik wel:", 
    "Ik zie een tweet die je kunt factchecken:"]

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
    phrase = random.choice(slack_intro_phrases)
        
    slack_json = {
        'text': f"{url}\n{phrase}"
    }
    
    # r = requests.post(slack_webhook_url, json=slack_json)
    # p = requests.post(statesman_webhook_url, json=slack_json)
    v = requests.post(slack_webhook_url, json=slack_json)
    
    status = {
        # 'quartz': r.status_code,
        # 'statesman': p.status_code,
        'vrtnws': v.status_code
    }
    
    return status

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

@app.route('/tweetcheck', methods=['POST'])
async def tweetcheck(request):
    incoming_json = await request.json()
    print(f"JSON is: {incoming_json}")
    
    analyze_text = incoming_json["textField"]
    prediction = learn.predict(analyze_text)[0]
    print(f"PREDICTION ^^^ is: {prediction}")
    
    ## Slack if prediction is True and it's NOT a retweet
    slack_says = "unsent"
    is_retweet = re.search("^RT ", analyze_text)
    if (str(prediction) is "True") and (is_retweet is None):
        slack_says = slack_this(prediction, incoming_json["tweetLink"])
    
    return JSONResponse({'result': str(prediction), 'slack_status': slack_says})

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
