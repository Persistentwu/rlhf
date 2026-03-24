from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "../output_sft/qwen_sft_final"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

# 准备对话历史和待比较的回答
conversation_history = """User: 看过七宗罪这部电影吗？
Assistant: 看过呀，这是1995年上映的影片。
User: 知道这是一部什么类型的影片吗？
Assistant: 这是一部剧情片，知道这是在哪里制片的吗？
User: 制片地区在美国，还记得编剧是谁吗？
Assistant: 编剧是安德鲁·凯文·沃克，影片讲述的是什么故事啊？
User: 该片以连环杀人案件为线索，从警员的视角出发，讲述了"七宗罪"系列谋杀案的故事。
Assistant: 影片获奖了吗？
User: 是啊，1996年该片获得了第5届MTV电影奖最佳影片等奖项，知道导演是谁吗？"""

answers_to_compare = {
    "answer1": "这是由大卫·芬奇执导的，听说过他吗？",
    "answer2": "这是由克里斯托弗·诺兰执导的，听说过他吗？",
    "answer3": "我最近在学做意大利面，番茄酱的配方很重要。",
    "answer4": "导演是 大卫·芬奇 啊 听说过 他 吗 这个 电影 的 奖项 很多。",
    "answer5": "哦。"
}

# 准备模型输入
prompt = "比较下列回答中哪句更好，并说明理由：\n\n"
for key, answer in answers_to_compare.items():
    prompt += f"{key}: {answer}\n"

messages = [
    {"role": "system", "content": "你是一个专业的对话评估助手，能够比较不同回答的质量。并给出打分，打分范围是0~10分，分数越大越好。"},
    {"role": "user", "content": f"以下是历史对话信息：\n{conversation_history}\n\n{prompt}"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)