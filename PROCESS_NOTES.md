# Process notes for Checkable Tweets

## Summary

Based on this course: https://course.fast.ai/


## Setup

I used this: https://course.fast.ai/start_aws.html

## Return to work

It's here for AWS: https://course.fast.ai/update_aws.html

Key lines:

```
ssh -L8888:localhost:8888 ubuntu@IP_ADDRESS
jupyter notebook
```

Then copy-paste the whole url provided into a browser:

```
http://localhost:8888/?token=232c5c...
```

- Click into `course-v3`
- Then `nbs` (for notebooks)
- Then `dl1` (for deep learning 1)

Remember to always make a copy of the notebook you want to tinker with.

## Moving files from local laptop to running server

From the laptop:

```
ssh-add .ssh/ai_studio.pem 
cd the_directory_im_working_in
scp filename ubuntu@IP_ADDRESS:data

```

## Moving them back to the laptop

From the laptop:

```
scp ubuntu@IP_ADDRESS:data/subdirectory/filename filename
```

## Adding an IFTTT function to track #txlege tweets

https://ifttt.com/applets/99922482d-if-new-txlege-tweet-add-row-to-google-drive-spreadsheet

## Let's make a lambda function!

Following this plan: https://course.fast.ai/deployment_aws_lambda.html

I'm working from [this fastai guide](https://course.fast.ai/deployment_aws_lambda.html). 

Note, though, that I'm really working on two computers: **My laptop** and an **Amazon EC2 instance** where I train my models using jupyter notebook. (For more on how to set that up, [see here](https://course.fast.ai/start_aws.html).

I already have the AWS command-line interface (CLI), the SAM CLI, and Docker on my laptop ... and it already has permission to do things on AWS. I don't want to go through the hassle of setting that all up again on my EC2 instance, So I'm going to split the tasks in the fastai guide.

### Womp womp ... can't be done yet

Turns out the lambda deployment above works wonderfully for images ... but not for text-classification models. This mainly appears to be a problem with the fact you can't load all of fast.ai onto a lambda function _AND_ that the workaround (turning it into a TorchScript JIT model) doesn't work for the structure of a language model.

And folks have tried, according to the [lambda discussion on the fastai forums](https://forums.fast.ai/t/deployment-platform-aws-lambda/39845). Definitely going to watch that space.

## Let's use Render

[Render.com](https://render.com) is a new service that lets you deploy apps cloud, and founder Anurag Goel has been [working closely with fastai](https://forums.fast.ai/t/deployment-platform-render/33953) on this (and other projects).

There's an example for using Render to process images [here](https://course.fast.ai/deployment_render.html) and I was excited to see that Raymond-Wu [successfully modified the code](https://forums.fast.ai/t/solved-trying-to-deploy-imdb-on-render/42043) to deploy the fastai IMDB text-classification model to Render. He [posted the code on Github](https://github.com/RaymondDashWu/nlp_classifier_restart2). 

[I've forked Raymond's here](https://github.com/jkeefe/nlp-classifier-fastai-render), and will be working from there and from a fork of the [Render starter app](https://github.com/render-examples/fastai-v3) -- which I think may be more current. Render draws right from your repo whenever you push to the `master` branch. (I chose to grant Render permissions to my repos on a case-by-case basis instead of giving it access to all of my repos, for what it's worth.)

While Render is free for static sites (which is cool), this implementation requires Docker, which means I have to subscribe to the $5/month version. 

[Image here]

I called my project `aistudio-nlp-tweets` ... which gave me this url: https://aistudio-nlp-tweets.onrender.com/

Then it installed the package and got it up and running. 

### Exporting the model

Back in my jupyter notebook, I did:

```
learn.export()
```

which saved the data, transforms, weights, etc into a file called `export.pkl` in my learn.path directory.

Copied that to my local machine:

```
cd Downloads
scp ubuntu@52.4.42.129:data/tweets_ams/export.pkl export.pkl
```

Then copied that to my S3 bucket.

```
cd Downloads
aws s3 cp export.pkl s3://qz-aistudio-public/checkable-tweets/export.pkl --acl public-read
```

Then in the S3 console got the URL: `https://s3.amazonaws.com/qz-aistudio-public/checkable-tweets/export.pkl`

Put that into the Render code.

Changed the classifications in the code.

### Troubleshooting

#### Round one (this didn't work)

- Tried starting from scratch with the [example instructions here](https://course.fast.ai/deployment_render.html#fork-the-starter-app-on-github).
- Had problem analyzing page, got: `AttributeError: 'Conv2d' object has no attribute 'padding_mode'`
- Saw in the forums that people had had this problem when the pytorch version of the model building (ie, in my notebook) didn't match the version in the Render requirements.
- Checked the one in my notebook (via the EC2 instance) like this:

```
python -c "import torch; print(torch.__version__)"
```

Which returned: `1.0.1.post2`

In the requirements, I see that it downloads what looks like a newer version, `1.1.0`: `https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whlfastai==1.0.52`

On this [pytorch install page](https://pytorch.org/get-started/locally/), I found this way to install:

```
# Python 3.7
pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl
pip3 install torchvision
```

... and the file name matches my `1.0.1.post2` version.

So replaced the Pytorch line in my `requirements.txt` with `https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl`

Sigh. Now I'm getting `AttributeError: 'ReLU' object has no attribute 'threshold'`

Wait. This is *the original* example code. And it's not working. Hmmm.

#### Round two

Looking at the _original_ repo (not my fork), I see that he's changed the `requirements.txt` file. Ah! 

So [found instructions](https://help.github.com/en/articles/merging-an-upstream-repository-into-your-fork) on how to update my fork to the original code.

```
git checkout master
git pull https://github.com/render-examples/fastai-v3.git master
```

- fixed the conflicts
- committed the merge
- did `git push origin master`

That triggered a new deploy "event" in the Render dashboard.

Waited a few minutes for it to install (click on the "Deploy" link in the events to watch the logs)

Worked! I can test for teddy bears at: https://aistudio-images.onrender.com/

(just like the demo at https://fastai-v3.onrender.com/ )

OK, now I'm going to copy all the files locally from my working image-recognition repo (`render-starter-app-fastai-v3`) to a branh of my broken text-classificaiton repo (`nlp-classifier-fastai-render`), being careful not to overwrite the `.git` file in the destination repo.

```
cd nlp-classifier-fastai-render/
git branch image-example
git checkout image-example

cd ../render-starter-app-fastai-v3/
cp .dockerignore ../nlp-classifier-fastai-render/
cp .gitignore ../nlp-classifier-fastai-render/
cp Dockerfile ../nlp-classifier-fastai-render/
cp requirements.txt ../nlp-classifier-fastai-render/
cp -r app ../nlp-classifier-fastai-render/

cd ../nlp-classifier-fastai-render/
```

- added the files to the branch
- committed them to the branch
- merged them up to master

Here's the diff of the two versions, which will be helpful in making the nlp version: https://github.com/jkeefe/nlp-classifier-fastai-render/pull/1/files

Merge created a new event in the Render web service for my nlp classifier. But really this should just be a working version of the image-testing example.

That works! 

[image here]

At https://aistudio-images.onrender.com/

### Modifying Sample for NLP

OK! I'm essentially where I would be had I _started_ in this repo with the updated example image code.

Next, I'm going to modify it based on Raymond's mod ...

- started a new branch off `master` called `build-classifier` 
- started with `view/index.html` page, which I copied over from Raymond's code.
- same for the `static/client.js` file
- in `server.py` ...
    - updated the model url to https://s3.amazonaws.com/qz-aistudio-public/checkable-tweets/export.pkl
    - changed the classes to `classes = ['True', 'False']`
    - changed the steps in the `/analyze` route
- Merged this all to `master`, which triggers a rebuild in Render

OK, that built and deployed! But getting an error on the processing.

Learning about starlette requests ... https://www.starlette.io/requests/

... that led me to `await request.json()` 

Got this working, tested with Postman 2:

Endpoint: https://aistudio-nlp-tweets.onrender.com/analyze
Content-Type: `application/json`
Body: `{"textField": "Dan Patrick didn't fulfill his promise to give teachers an average $10,000 in salary raises."}`

## Wiring it all up

### IFTTT as the detector

I set up an IFTTT applet to await a tweet containing `#txlege` and then send a "webhook"

(Took lots of screenshots; See July 4, 2019 11 am)

Note that the IFTTT instructions are _incorrect_ -- you need to escape any text ingredient (in my case, the tweet text) with _triple_ angle brackets. The ingredients themselves are surrounded with double curly brackets. So the final thing looks like this: `<<<{{TextIngredient}>>>`

For the URL, I put my Render app's url and endpoint.

I'm using POST ... and also `application/json`

### True (but RT)

From the Render logs:

```
Jul 4 04:32:30 PM  JSON is: {'textField': "RT @TexasNORML: #Veteran overdose deaths from opioids are rising. #Texas has the second highest veteran population in the nation (1.5M), committing suicide at more than twice the state rate. \n\nDespite evidence #marijuana is medicine, #txlege didn't include #PTSD.\n\nhttps://t.co/L3ZthtzxWZ https://t.co/wlvRa5N6BP", 'tweetLink': 'http://twitter.com/SpiderguyBen/status/1146871001587834880'}
Jul 4 04:32:30 PM  PREDICTION ^^^ is: True
```




