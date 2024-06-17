import os
from groq import Groq


client = Groq(api_key=os.environ.get("GROQ_API_KEY"), )


system_prompt = {
  
    "role": "system",
    "content":
    "You are a helpful assistant. You reply with very short answers."
}

chat_history = [system_prompt]

while True:

  user_input = input("You: ")

  chat_history.append({"role": "user", "content": user_input})

  # 可以選擇你要的 model
  response = client.chat.completions.create(model="llama3-70b-8192",
                                            messages=chat_history,
                                            max_tokens=100,
                                            temperature=1.2)
  # 儲存每次的對話到 chat 中
  chat_history.append({
      "role": "assistant",
      "content": response.choices[0].message.content
  })


  print("groq:", response.choices[0].message.content)
