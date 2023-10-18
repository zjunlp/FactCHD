import time

import openai


def get_response(prompt, model="gpt-3.5-turbo-0301", temperature=0.7, max_tokens=512, api_type="azure"):
    if api_type == "azure":
        openai.api_type = "azure"
        openai.api_base = "Your Azure OpenAI resource's endpoint value."
        openai.api_key = "Your Azure OpenAI resource's api key"
        if model == "gpt-3.5-turbo-0301":
            openai.api_version = "2023-03-15-preview"
            model = "gpt-35-turbo-0301"
            response = openai.ChatCompletion.create(
                max_tokens=max_tokens,
                temperature=temperature,
                engine=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            return response.choices[0].message.content
        elif model == "text-davinci-003":
            openai.api_version = "2022-12-01"
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            return response.choices[0].text
    elif api_type == "openai":
        if model == "gpt-3.5-turbo-0301":
            openai.api_key = ""
            response = openai.ChatCompletion.create(
                # max_tokens=max_tokens,
                temperature=temperature,
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            return response.choices[0].message.content
        elif model == "text-davinci-003":
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            return response.choices[0].text

    return None


def get_response_retry(prompt, model="gpt-3.5-turbo-0301", temperature=0.7, max_tokens=512, retry=2, api_type="openai"):
    i = 1
    while i <= retry:
        try:
            return get_response(prompt, model, temperature, max_tokens, api_type)
        except Exception as e:
            print("retry:" + str(i))
            print(e)
            if str(e).startswith("The response was filtered due") or str(e).startswith("content"):
                return None
            time.sleep(30)
            i += 1
    return None


if __name__ == "__main__":
    print(get_response("你好，你是？", model="gpt-3.5-turbo-0301", api_type="openai"))
