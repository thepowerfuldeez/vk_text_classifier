# vk_text_classifier
Neural network classifier of vk (or even facebook) profiles based on texts corpora.
Files needed to run: `dump.rdb categories.json fb_dump.json vk_texts_classifier.h5 vectorizer.p margins.json`
(all files go into assets/ subdir)

## Usage:
#### Setting up config.py
```
VK_TOKEN = VK_TOKEN
FB_TOKEN = FB_TOKEN
```

#### Using API:
```
docker build -t vk_text_classifier .

docker-compose up
```

#### API Reference
```json
POST /get_result
json={
	"name": 'Имя человека',
	"user_vk": VK_ID,
	"user_fb": FB_NAME,
	"verbose": True/False (default: False)
}
```

**Example**:
```python
>>> requests.post("http://78.155.197.212:9999/get_result", json={"name": "Georgiy", "user_vk": 134070307}).json()
{'name': 'Georgiy', 'results': ['Юриспруденция', 'Исследования и разработки', 'Благотворительность', 'Инновации и модернизация']}
```
