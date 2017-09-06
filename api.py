import flask
import json
from util import ResultClass

app = flask.Flask(__name__)
result = ResultClass()
labels = ["Искусство", "Политика", "Финансы", "Стратегическое управление", "Юриспруденция", "Исследования и разработки",
          "Промышленность", "Образование", "Благотворительность", "Здравоохранение", "Сельское хозяйство",
          "Государственное управление", "Реклама и маркетинг", "Инновации и модернизация", "Безопасность",
          "Военное дело", "Корпоративное управление", "Социальная защита", "Строительство", "Предпринимательство",
          "Спорт", "Инвестиции"]


@app.route("/")
def index():
    return "Welcome", 200


@app.route("/get_result", methods=["POST"])
def get_result():
    """
    POST http://0.0.0.0:9999/get_result
    json={"name": NAME, "user_vk": VK_ID, "user_fb": "FB_PAGE}
    :return:
    """
    data = flask.request.get_json()
    name = data.get('name', "")
    user_vk = data.get('user_vk')
    user_fb = data.get('user_fb')
    verdict = result.get_result(user_vk, user_fb)
    norm_names = dict(zip([a[0] for a in verdict], labels))
    results = []
    for col, value in verdict:
        results.append({"name": norm_names[col], "value": float(value)})
    result.texts = []
    return app.response_class(
        response=json.dumps({"name": name, "results": results}),
        status=200,
        mimetype='application/json'
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999)
