from court import Judge, Jury
from utils import launch, TestSetGenerator


def main():
    generator = TestSetGenerator()
    questions = generator.generate_questions()
    answers = generator.generate_answers()

    launch()

    judge = Judge()
    jury = Jury()

    reply = jury.chat(messages=questions)
    messages = answers + [{"role": "user", "content": "[START INPUT]" + reply + "[END INPUT]"}]
    rep = judge.chat(messages=messages)

    print(rep)


if __name__ == "__main__":
    main()
