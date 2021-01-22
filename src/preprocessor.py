import re

class RegexPreprocessor():
    '''데이터 정규식 전처리기'''

    def __init__(self):
        self.cpu_regex = ""
        self.vga_regex = ""
        self.mb_regex = ""
        self.ram_regex = ""
        self.ssd_regex = ""
        self.power_regex = ""
    
    def cpu(self, text):
        flag = "AMD"
        for check in ["INTEL", "intel", "인텔"]:
            if check in text:
                flag = "INTEL"
        
        if flag == "INTEL":
            regex_result = re.findall("i\d|\d{4,5}KF|\d{4,5}K|\d{4,5}F|\d{4,5}X|\d{4,5}", text)
        else:
            regex_result = re.findall("라이젠\d|\d{4,5}X|\d{4,5}G", text)

        if regex_result:
            return " ".join(regex_result)
        else:
            None
    
    def vga(self, text):
        regex_result = re.findall("GAINWARD|이엠텍|MSI|ZOTAC|갤럭시|ASUS|GIGABYTE|PowerColor|리드텍|AFOX|AKiTiO|AMD|ARKTEK|ASRock|ATUM|AXLE|COLORFUL|EVGA|FORSA|HIS|INNO3D|MANLI|MAXSUN|NETSTOR|NVIDIA|PALIT|PNY|Razer|SAPPHIRE|SNK테크|SOYO|Sonnet|TAGER|XFX|레노버|매트록스|세컨드찬스|엠탑코리아|GTX \d{3,4} Ti|GTX \d{3,4}|RTX \d{3,4}|RTX \d{3,4}Ti|RX \d{3,4} XT|RX \d{3,4}", text)

        if regex_result:
            return " ".join(regex_result)
        else:
            None

    def mb(self, text):
        regex_result = re.findall("[A-Z]\d{2,3}\w+|ASRock|ASUS|MSI|GIGABYTE|ECS|AFOX|ASRock Rack|Arduino|BIOSTAR|COLORFUL|FOXCONN|JETWAY|Maxtang|Raspberry Pi|Supermicro|TYAN|디지탈그린텍|마이크로닉스|이엠텍|인텍앤컴퍼니|인텔|코코아팹", text)

        if regex_result:
            return " ".join(regex_result)
        else:
            None
    
    def ram(self, text):
        regex_result = re.findall("삼성전자|ADATA|G.SKILL|GeIL|ACPI|AFOX|AVEXIR|Antec|Apacer|CORSAIR|CYNEX|Dreamware|EKMEMORY|ESSENCORE|GIGABYTE|GLOWAY|GSMS|HP|INNO3D|KINGMAX|LANSON|OCPC|OLOy|PATRIOT|PNY|SK하이닉스|TeamGroup|Terabyte|V-Color|ZADAK|갤럭시|건평정보통신|디자인|마이크론|실리콘파워|써멀테이크|어드반|오존컴퍼니|이메이션|킹스톤|타무즈|트랜센드|\d{1,2}GB|\d{1,2}G", text)

        if regex_result:
            return " ".join(regex_result)
        else:
            None

    def ssd(self, text):
        regex_result = re.findall("삼성전자|마이크론|ADATA|Western Digital|ACPI|AFOX|ASUS|AVEXIR|Apacer|Axxen|BIOSTAR|BIWIN|BLUE-I|COLORFUL|COOLERTEC|CORSAIR|CRAFT|DATARAM|DIGIFAST|DIGISTOR|EAGET|EKMEMORY|ESSENCORE|EVERCOOL|EXASCEND|FOXCONN|Faspeed|GIGABYTE|GLOWAY|GeIL|GrinMax|HGST|HIKVISION|HP|ICY DOCK|IPT|JEYI|KINGMAX|Kim MiDi|Kimtigo|KingDian|KingSpec|Korlet|Lexar|Lite-On|Longsys|MAIWO|MARSHAL|MK2|MUSHKIN|MiSD|MyDigitalSSD|MySSD|NCTOP|NOFAN|Netac|OCPC|OCZ SS|ORICO|OWC|PALIT|PATRIOT|PHINOCOM|PNY|Plextor|RAMIS|RiTEK|SK하이닉스|SONY|STARSWAY|STCOM|SUNEAST|Sandisk|Seagate|SilverStone|Supertalent|Synology|TCSUNBOW|TOPDISK|TeamGroup|Toshiba|UNITEK|Union Memory|VIA|Vaseky|VisionTek|ZOTAC|innoDisk|\d{3}GB|\dTB|\d{3}gb|\dtb|\d{3}G|\dT|\d{3}g|\dt", text)

        if regex_result:
            return " ".join(regex_result)
        else:
            None
    
    def power(self, text):
        regex_result = re.findall("\d{3,4}W|\d{3,4}w|\d{3,4}", text)

        if regex_result:
            return regex_result[0]
        else:
            return None
