from src.judge import Judge
from utils.startup import launch


def main():
    launch()
    message = [{"role": "user", "content": "Why is the sky blue?"}]

    judge = Judge()
    reply = judge.judge(messages=message)
    print(reply)


if __name__ == "__main__":
    main()
