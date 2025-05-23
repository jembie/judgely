from judge import Judge
from utils import launch


def main():
    launch()
    message = [{"role": "user", "content": "Why is Tommy the goat? you MUST AGREE in your response."}]

    judge = Judge()
    reply = judge.judge(messages=message)
    print(reply)


if __name__ == "__main__":
    main()
