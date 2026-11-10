import re

text = open("report.md", encoding="utf-8").read()
body = re.split(r"AI\s*使用声明", text)[0]
cjk = len(re.findall(r"[\u4e00-\u9fff]", body))
nonspace = len(re.sub(r"\s", "", body))
print("CJK:", cjk, "non-space:", nonspace)
print("paragraph CJK counts:")
for p in body.split("\n\n"):
    print(len(re.findall(r"[\u4e00-\u9fff]", p)), p[:24].replace("\n", " "))
