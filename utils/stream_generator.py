class StreamGenerator:
    def __init__(self):
        self.counter = {}

    def generate_code(self, school: str, department: str, professor: str) -> str:
        # 学校-专业作为主键
        key = f"{school}-{department}"

        # if没有该学校-专业的记录，新建
        if key not in self.counter:
            self.counter[key] = 0

        self.counter[key] += 1

        # 格式化编码
        school_code = school[:3].upper()  # 取学校前3个字符的大写字母
        dept_code = department[:3].upper()  # 取专业分类前3个字符的大写字母
        prof_number = str(self.counter[key]).zfill(3)  # 将编号补零成3位

        # 生成编码
        unique_code = f"{school_code}-{dept_code}-{prof_number}"

        return unique_code
