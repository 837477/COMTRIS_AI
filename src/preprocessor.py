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
        self.ram_clock = {
            "25600": "3200",
            "24000": "3000",
            "23400": "2933",
            "21300": "2666",
            "19200": "2400",
            "38400": "4800",
            "36800": "4600",
            "36000": "4500",
            "35200": "4400",
            "34400": "4300",
            "34100": "4266",
            "33000": "4133",
            "32000": "4000",
            "30900": "3866",
            "30400": "3800",
            "28800": "3600",
            "27700": "3466",
            "27200": "3400",
            "26600": "3333",
            "22400": "2800",
            "17000": "2133",
            "22400": "2800",
            "21300": "2666",
            "19200": "2400",
            "17000": "2133",
            "14900": "1866",
            "12800": "1600",
            "10600": "1333",
            "10000": "1250",
            "8500" : "1066"
        }
    
    def cpu(self, text):
        flag = "AMD"
        for check in ["INTEL", "intel", "인텔"]:
            if check in text:
                flag = "INTEL"
        
        if flag == "INTEL":
            regex_result = re.findall("\d{4,5}KF|\d{4,5}K|\d{4,5}F|\d{4,5}X|\d{4,5}", text)
        else:
            regex_result = re.findall("\d{4,5}X|\d{4,5}G|\d{4,5}", text)

        if regex_result:
            return " ".join(regex_result)
        else:
            None
    
    def vga(self, text):
        # brand = re.findall("GAINWARD|이엠텍|MSI|ZOTAC|갤럭시|ASUS|GIGABYTE|PowerColor|리드텍|AFOX|AKiTiO|AMD|ARKTEK|ASRock|ATUM|AXLE|COLORFUL|EVGA|FORSA|HIS|INNO3D|MANLI|MAXSUN|NETSTOR|NVIDIA|PALIT|PNY|Razer|SAPPHIRE|SNK테크|SOYO|Sonnet|TAGER|XFX|레노버|매트록스|세컨드찬스|엠탑코리아", text)
        chipset = re.findall("GTX\d{3,4}SUPER|GTX \d{3,4}SUPER|GTX\d{3,4} SUPER|GTX \d{3,4} SUPER|GTX\d{3,4}Ti|GTX \d{3,4}Ti|GTX\d{3,4} Ti|GTX \d{3,4} Ti|GTX\d{3,4}TI|GTX \d{3,4}TI|GTX\d{3,4} TI|GTX \d{3,4} TI|GTX\d{3,4}|GTX \d{3,4}|RTX\d{3,4}super|RTX \d{3,4}super|RTX\d{3,4} super|RTX \d{3,4} super|RTX\d{3,4}SUPER|RTX \d{3,4}SUPER|RTX\d{3,4} SUPER|RTX \d{3,4} SUPER|RTX\d{3,4}Ti|RTX \d{3,4}Ti|RTX\d{3,4} Ti|RTX \d{3,4} Ti|RTX\d{3,4}|RTX \d{3,4}|RX\d{3,4}XT|RX \d{3,4}XT|RX\d{3,4} XT|RX \d{3,4} XT|RX\d{3,4}|RX \d{3,4}", text)

        if (not chipset):
            return None
        
        return chipset[0].upper().replace(" ", "")


    def mb(self, text):
        # brand = re.findall("ASRock|ASUS|MSI|GIGABYTE|ECS|AFOX|ASRock Rack|Arduino|BIOSTAR|COLORFUL|FOXCONN|JETWAY|Maxtang|Raspberry Pi|Supermicro|TYAN|디지탈그린텍|마이크로닉스|이엠텍|인텍앤컴퍼니|인텔|코코아팹", text)
        chipset = re.findall("[A-Z]\d{2,3}\w+", text)

        if (not chipset):
            return None
        
        return chipset[0]
    
    def ram(self, text):
        # brand = re.findall("삼성전자|ADATA|G.SKILL|GeIL|ACPI|AFOX|AVEXIR|Antec|Apacer|CORSAIR|CYNEX|Dreamware|EKMEMORY|ESSENCORE|GIGABYTE|GLOWAY|GSMS|HP|INNO3D|KINGMAX|LANSON|OCPC|OLOy|PATRIOT|PNY|SK하이닉스|TeamGroup|Terabyte|V-Color|ZADAK|갤럭시|건평정보통신|디자인|마이크론|실리콘파워|써멀테이크|어드반|오존컴퍼니|이메이션|킹스톤|타무즈|트랜센드", text)
        chipset = re.findall("\d{5}|\d{4}", text)
        volume = re.findall("\d{1,2}GB|\d{1,2}gb|\d{1,2}G|\d{1,2}g", text)

        if (not chipset) or (not volume):
            return None
        
        # 칩셋 재가공
        if len(chipset[0]) == 5:
            chipset[0] = self.ram_clock[chipset[0]]

        # 용량 재가공
        if len(volume) >= 2:
            for idx, value in enumerate(volume):
                volume[idx] = value.replace("GB", "")
                volume[idx] = volume[idx].replace("G", "")
                volume[idx] = int(volume[idx])
            volume[0] = str(max(volume)) + "GB"

        return chipset[0] + " " + volume[0]

    def ssd(self, text):
        # brand = re.findall("삼성전자|마이크론|ADATA|Western Digital|ACPI|AFOX|ASUS|AVEXIR|Apacer|Axxen|BIOSTAR|BIWIN|BLUE-I|COLORFUL|COOLERTEC|CORSAIR|CRAFT|DATARAM|DIGIFAST|DIGISTOR|EAGET|EKMEMORY|ESSENCORE|EVERCOOL|EXASCEND|FOXCONN|Faspeed|GIGABYTE|GLOWAY|GeIL|GrinMax|HGST|HIKVISION|HP|ICY DOCK|IPT|JEYI|KINGMAX|Kim MiDi|Kimtigo|KingDian|KingSpec|Korlet|Lexar|Lite-On|Longsys|MAIWO|MARSHAL|MK2|MUSHKIN|MiSD|MyDigitalSSD|MySSD|NCTOP|NOFAN|Netac|OCPC|OCZ SS|ORICO|OWC|PALIT|PATRIOT|PHINOCOM|PNY|Plextor|RAMIS|RiTEK|SK하이닉스|SONY|STARSWAY|STCOM|SUNEAST|Sandisk|Seagate|SilverStone|Supertalent|Synology|TCSUNBOW|TOPDISK|TeamGroup|Toshiba|UNITEK|Union Memory|VIA|Vaseky|VisionTek|ZOTAC|innoDisk", text)
        volume = re.findall("\d{3}GB|\dTB|\d{3}gb|\dtb|\d{3}G|\dT|\d{3}g|\dt", text)

        if (not volume):
            return None
        
        return volume[0].upper()
    
    def power(self, text):
        regex_result = re.findall("\d{3,4}W|\d{3,4}w|\d{3,4}", text)

        if not regex_result:
            return None

        regex_result[0] = regex_result[0].upper()

        if regex_result[0][-1] != "W":
            regex_result[0] = regex_result[0] + "W"
        
        return regex_result[0]
