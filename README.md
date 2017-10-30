# vk_text_classifier
Neural network classifier of vk (or even facebook) profiles based on texts corpora.
Files needed to run: `dump.rdb vk_texts_classifier.h5 vectorizer.p margins.json`
(all files go into assets/ subdir)


#### Using API:
```
docker build -t vk_text_classifier .

docker-compose up -d
```

#### API Reference
##### POST /get_result
```json
{
	"name": "Georgiy", (not necessary)
	"user_vk": VK_TOKEN or VK_ID,
	"user_fb": FB_TOKEN or FB_ID,
	"verbose": True/False (default: False)
}
```
* Vk id may be used if there's record in database, in all other cases you should use token to get result.

**Example**:
```python
>>> requests.post("http://78.155.197.212:9999/get_result", json={"name": "Georgiy", "user_vk": 134070307}).json()
{'name': 'Georgiy', 'results': ['Юриспруденция', 'Исследования и разработки', 'Благотворительность', 'Инновации и модернизация']}
```
