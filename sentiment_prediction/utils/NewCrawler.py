import requests
import pandas as pd

titles={'title':[],'sentiment':[]}
print('Crawl pocinje')

sess = requests.Session()

for i in range(60):
    response=sess.get('https://api.pushshift.io/reddit/search/submission/?subreddit=upliftingnews&sort=desc&sort_type=score&before='+str(i*30)+'d&after='+str((i+1)*30)+'d&size=250')
    json=response.json()
    data=json['data']
    for k in range(250):
        try:
            titles['title'].append(data[k]['title'])
            titles['sentiment'].append(0)
        except:
            pass
    if i%10==0:
        print("Crawlano %d/%d dana"%(i*30,60*30))
for i in range(60):
    response=sess.get('https://api.pushshift.io/reddit/search/submission/?subreddit=morbidreality&sort=desc&sort_type=score&before='+str(i*30)+'d&after='+str((i+1)*30)+'d&size=250')
    json=response.json()
    data=json['data']
    for k in range(250):
        try:
            titles['title'].append(data[k]['title'])
            titles['sentiment'].append(1)
        except:
            pass
    if i%10==0:
        print("Crawlano %d/%d dana"%(i*30,60*30))
print('Crawl gotov')

topics_data = pd.DataFrame(titles)
topics_data.to_csv('kita.csv', index=False)