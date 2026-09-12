"""Create a deterministic, entirely fictional 306-note graph showcase.

Usage: python examples/generate_graph_showcase.py .local/graph-showcase-vault
The destination must not already exist. No original notes are read.
"""

import sys
from datetime import date, timedelta
from pathlib import Path


def generate(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    topics = ['阅读', '写作', '散步', '专注', '学习', '选择', '生活', '观察']
    hub = '把日子写成线索'
    companion = '给想法留一条回来的路'
    names = [[f'{topic}手记-{i + 1:02d}' for i in range(38)] for topic in topics]

    def write(title: str, body: str, links: list[str], day: date) -> None:
        text = f'---\ntitle: {title}\ncreated: {day.isoformat()}\ntags: [合成展示]\n---\n\n{body}\n\n'
        text += ' '.join(f'[[{link}]]' for link in dict.fromkeys(links)) + '\n'
        (destination / f'{title}.md').write_text(text, encoding='utf-8')

    write(hub, '重读这些手记时，我发现，散落的片段也能彼此照亮。\n\n一段阅读、一场散步、一次尚未想清楚的选择，都可以先留下来。联系慢慢浮现，解释不必急着完成。',
          [companion, *(group[0] for group in names)], date(2025, 5, 18))
    write(companion, '过去总希望记下一个确定的答案，现在也愿意保存一个尚未回答的问题。\n\n让每条线索有出处，让每次重读有余地。下一次回来看，也允许自己有不同的理解。',
          [hub, names[0][0], names[1][0]], date(2025, 6, 9))
    for group_index, group in enumerate(names):
        for index, title in enumerate(group):
            links = [group[(index + 1) % 38], group[(index + 5) % 38]]
            if index % 4 == 0:
                links.append(hub)
            if index % 7 == 0:
                links.append(names[(group_index + 1) % len(names)][index])
            write(title, f'这是用于界面展示的虚构{topics[group_index]}片段。\n\n今天留下一点观察，下一次回读时，再看看它与其他记录有什么联系。',
                  links, date(2023, 1, 1) + timedelta(days=group_index * 91 + index * 3))
    print(f'Created 306 synthetic notes in {destination}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    generate(Path(sys.argv[1]))
