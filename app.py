from flask import Flask, render_template, request, redirect, jsonify
from datetime import datetime
import pickle
import pandas as pd
import os
import numpy as np

with open('/home/ubuntu/flaskproject/model.pckl', 'rb') as fin:
	prophet_model = pickle.load(fin)

#df = pd.read_csv('/home/ubuntu/flaskproject/CottonData.csv')
#dummy=df.drop('Price',axis='columns')

state={'Andhra Pradesh': 0, 'Gujarat': 1, 'Haryana': 2, 'Karnataka': 3, 'Madhya Pradesh': 4, 'Maharashtra': 5, 'Meghalaya': 6, 'Odisha': 7, 'Pondicherry': 8, 'Punjab': 9, 'Rajasthan': 10, 'Tamil Nadu': 11, 'Telangana': 12, 'Uttar Pradesh': 13}
district={'Anantapur': 11, 'Cuddapah': 41, 'East Godavari': 49, 'Guntur': 62, 'Krishna': 91, 'Kurnool': 93, 'Prakasam': 123, 'Srikakulam': 142, 'Vijayanagaram': 154, 'Visakhapatnam': 157, 'West Godavari': 160, 'Ahmedabad': 1, 'Amreli': 9, 'Banaskanth': 16, 'Bharuch': 22, 'Bhavnagar': 24, 'Gandhinagar': 58, 'Jamnagar': 72, 'Junagarh': 77, 'Kachchh': 78, 'Kheda': 87, 'Mehsana': 102, 'Morbi': 104, 'Narmada': 114, 'Panchmahals': 118, 'Patan': 120, 'Rajkot': 126, 'Sabarkantha': 132, 'Surat': 143, 'Surendranagar': 144, 'Vadodara(Baroda)': 151, 'Bhiwani': 26, 'Faridabad': 51, 'Fatehabad': 53, 'Hissar': 69, 'Jhajar': 74, 'Jind': 75, 'Kaithal': 79, 'Mewat': 103, 'Rohtak': 131, 'Sirsa': 138, 'Bagalkot': 15, 'Belgaum': 19, 'Bellary': 20, 'Bijapur': 27, 'Chamrajnagar': 33, 'Chikmagalur': 36, 'Chitradurga': 37, 'Davangere': 42, 'Dharwad': 46, 'Gadag': 55, 'Gulbarga': 61, 'Hassan': 65, 'Haveri': 67, 'Karwar(Uttar Kannad)': 83, 'Kolar': 88, 'Koppal': 89, 'Mandya': 98, 'Mysore': 106, 'Raichur': 125, 'Shimoga': 136, 'Alirajpur': 6, 'Badwani': 14, 'Burhanpur': 32, 'Chhindwara': 35, 'Dewas': 43, 'Dhar': 44, 'Harda': 64, 'Jhabua': 73, 'Khandwa': 85, 'Khargone': 86, 'Ratlam': 129, 'Ahmednagar': 2, 'Akola': 4, 'Amarawati': 8, 'Aurangabad': 13, 'Beed': 18, 'Buldhana': 31, 'Chandrapur': 34, 'Dhule': 47, 'Gadchiroli': 56, 'Hingoli': 68, 'Jalana': 70, 'Jalgaon': 71, 'Nagpur': 109, 'Nanded': 112, 'Nandurbar': 113, 'Parbhani': 119, 'Sangli': 134, 'Wardha': 159, 'Yavatmal': 162, 'South West Khasi Hills': 141, 'West Khasi Hills': 161, 'Bolangir': 29, 'Gajapati': 57, 'Ganjam': 60, 'Kalahandi': 80, 'Koraput': 90, 'Rayagada': 130, 'Pondicherry': 121, 'Barnala': 17, 'Bhatinda': 23, 'Faridkot': 52, 'Fazilka': 54, 'Ludhiana': 94, 'Mansa': 99, 'Muktsar': 105, 'Sangrur': 135, 'Ajmer': 3, 'Alwar': 7, 'Bharatpur': 21, 'Bhilwara': 25, 'Ganganagar': 59, 'Hanumangarh': 63, 'Jodhpur': 76, 'Nagaur': 108, 'Pali': 116, 'Ariyalur': 12, 'Coimbatore': 39, 'Cuddalore': 40, 'Dharmapuri': 45, 'Dindigul': 48, 'Erode': 50, 'Krishnagiri': 92, 'Madurai': 95, 'Nagapattinam': 107, 'Namakkal': 111, 'Ramanathapuram': 127, 'Salem': 133, 'Theni': 146, 'Thirunelveli': 147, 'Thiruvarur': 148, 'Tuticorin': 150, 'Vellore': 153, 'Villupuram': 155, 'Virudhunagar': 156, 'Adilabad': 0, 'Karimnagar': 82, 'Khammam': 84, 'Mahbubnagar': 96, 'Medak': 101, 'Nalgonda': 110, 'Nizamabad': 115, 'Ranga Reddy Dist.': 128, 'Warangal': 158, 'Hathras': 66, 'Tumkur': 149, 'Vashim': 152, 'Karaikal': 81, 'Palwal': 117, 'Pune': 124, 'Sonepur': 140, 'Chittorgarh': 38, 'Aligarh': 5, 'Anand': 10, 'Mahendragarh-Narnaul': 97, 'Bikaner': 28, 'Sikar': 137, 'Sivaganga': 139, 'Thanjavur': 145, 'Mathura': 100, 'Botad': 30, 'Porbandar': 122}
market={'Gooti': 202, 'Jammalamadugu': 263, 'Kamalapuram': 298, 'Mydukur': 431, 'Proddatur': 510, 'Pithapuram': 503, 'Chilakaluripet': 124, 'Krosuru': 358, 'Macharla': 380, 'Pidugurala(Palnadu)': 500, 'Sattenapalli': 564, 'Tadikonda': 616, 'Vinukonda': 679, 'Jaggayyapeta': 247, 'Kanchekacherla': 302, 'Mylavaram': 432, 'Nandigama': 442, 'Nuzvid': 468, 'Adoni': 5, 'Allagadda': 14, 'Alur': 15, 'Banaganapalli': 49, 'Dhone': 154, 'Koilkunta': 343, 'Yemmiganur': 697, 'Darsi': 136, 'Markapur': 411, 'Martur': 412, 'Parchur': 486, 'Amadalavalasa': 17, 'Hiramandalam': 231, 'Palakonda': 473, 'Pathapatnam': 494, 'Rajam': 519, 'Bobbili': 97, 'Gajapathinagaram': 176, 'Kurupam': 365, 'Parvathipuram': 492, 'Saluru': 550, 'Visakhapatnam': 683, 'Chintalapudi': 127, 'Polavaram': 505, 'Dhandhuka': 146, 'Dholka': 152, 'Dholka(Koth)': 153, 'Ranpur': 535, 'Viramgam': 680, 'Amreli': 22, 'Babra': 38, 'Bagasara': 43, 'Damnagar': 135, 'Dhari': 148, 'Khambha': 326, 'Rajula': 524, 'Savarkundla': 568, 'Amirgadh': 20, 'Deesa(Bhildi)': 141, 'Diyodar': 162, 'Palanpur': 475, 'Amod': 21, 'Hasot': 223, 'Jambusar': 260, 'Jambusar(Kaavi)': 261, 'Jhagadiya': 275, 'Valia': 661, 'Valia(Nethrang)': 662, 'Bhavnagar': 76, 'Botad(Bhabarkot)': 108, 'Botad(Haddad)': 109, 'Gadada': 172, 'Gariyadar': 186, 'Mahuva(Station Road)': 388, 'Palitana': 476, 'Taleja': 617, 'Mansa': 404, 'Bhanvad': 73, 'Dhrol': 156, 'Jam Jodhpur': 257, 'Jam Khambalia': 258, 'Jamnagar': 265, 'Kalawad': 294, 'Lalpur': 370, 'Bhesan': 78, 'Junagadh': 282, 'Kodinar(Dollasa)': 342, 'Manavdar': 399, 'Una': 649, 'Visavadar': 684, 'Anjar': 25, 'K.Mandvi': 284, 'Balasinor': 46, 'Kapadanj(Moti Jaher)': 304, 'Kapadvanj': 305, 'Virpur': 681, 'Becharaji': 64, 'Kadi': 287, 'Kadi(Kadi cotton Yard)': 288, 'Unava': 650, 'Vijapur(Gojjariya)': 674, 'Visnagar': 685, 'Vankaner': 665, 'Rajpipla': 523, 'Gogamba': 194, 'Harij': 221, 'Patan': 493, 'Siddhpur': 590, 'Dhoraji': 155, 'Gondal': 199, 'Jamkandorna': 262, 'Jasdan': 269, 'Jetpur(Dist.Rajkot)': 272, 'Morbi': 425, 'Rajkot': 522, 'Upleta': 651, 'Bayad': 60, 'Bayad(Demai)': 61, 'Bayad(Sadamba)': 62, 'Bhiloda': 82, 'Dhansura': 147, 'Himatnagar': 227, 'Idar': 243, 'Idar(Jadar)': 244, 'Khedbrahma': 334, 'Malpur': 396, 'Meghraj': 419, 'Modasa': 422, 'Modasa(Tintoi)': 423, 'Talod': 619, 'Vadali': 654, 'Kosamba(Vankal)': 351, 'Kosamba(Zangvav)': 352, 'Nizar': 464, 'Chotila': 131, 'Halvad': 215, 'Limdi': 374, 'Muli': 427, 'Sayala': 573, 'Vadhvan': 656, 'Bodeli': 98, 'Bodeli(Hadod)': 99, 'Bodeli(Kalediya)': 100, 'Bodeli(Modasar)': 101, 'Jepur Pavi(Chackak)': 271, 'Jetpur-Pavi': 273, 'Karjan': 311, 'Nasvadi': 456, 'Nasvadi(Thalkala)': 457, 'Savli': 569, 'Savli(Desar)': 570, 'Savli(Samlaya)': 571, 'Vagodiya': 658, 'Bhiwani': 83, 'Ch. Dadri': 114, 'Jui': 280, 'Loharu(Dighwa)': 378, 'Siwani': 602, 'Tosham': 644, 'Palwal': 477, 'Bhattu Kalan': 75, 'Fatehabad': 169, 'Jakhal': 253, 'Ratia': 536, 'Tohana': 643, 'Adampur': 3, 'Barwala(Hisar)': 56, 'Hansi': 216, 'Narnaund': 451, 'Uklana': 646, 'Beri': 67, 'Jullana': 281, 'Narwana': 455, 'Pillukhera': 502, 'Uchana': 645, 'Kalayat': 295, 'Hathin': 224, 'Meham': 420, 'Dabwali': 133, 'Ding': 161, 'Ellanabad': 166, 'kalanwali': 701, 'Sirsa': 601, 'Badami': 39, 'Bilagi': 92, 'Hungund': 238, 'Jamakhandi': 259, 'Athani': 36, 'Bailahongal': 45, 'Gokak': 196, 'Kudchi': 361, 'Ramdurga': 529, 'Sankeshwar': 558, 'Soundati': 604, 'Bellary': 65, 'H.B. Halli': 209, 'Hoovinahadagali': 236, 'Kottur': 356, 'Sirguppa': 599, 'Bijapur': 90, 'Sindagi': 593, 'Talikot': 618, 'Chamaraj Nagar': 115, 'Gundlupet': 207, 'Kollegal': 345, 'Bagepalli': 44, 'Kadur': 289, 'Tarikere': 622, 'Chitradurga': 129, 'Davangere': 139, 'Harappana Halli': 220, 'Honnali': 235, 'Annigeri': 26, 'Dharwar': 151, 'Hubli (Amaragol)': 237, 'Kalagategi': 291, 'Kundagol': 364, 'Gadag': 173, 'Laxmeshwar': 373, 'Nargunda': 448, 'Gulburga(Jhevargi)': 206, 'Shorapur': 587, 'Arasikere': 30, 'Belur': 66, 'Haveri': 225, 'Hirekerur': 232, 'Ranebennur': 533, 'Savanur': 567, 'Shiggauv': 583, 'Haliyala': 214, 'Mundgod': 430, 'Yellapur': 695, 'Chintamani': 128, 'Malur': 397, 'Gangavathi': 185, 'Yalburga': 692, 'Nagamangala': 433, 'Srirangapattana': 609, 'Hunsur': 239, 'K.R.Nagar': 286, 'Nanjangud': 445, 'Santhesargur': 559, 'T. Narasipura': 615, 'Lingasugur': 375, 'Manvi': 407, 'Raichur': 517, 'Sindhanur': 594, 'Bhadravathi': 70, 'Shikaripura': 584, 'Shimoga': 585, 'Alirajpur': 13, 'Jobat': 277, 'Anjad': 24, 'Badwani': 42, 'Balwadi': 48, 'Khetia': 335, 'Sendhwa': 577, 'Burhanpur': 112, 'Pandhurna': 480, 'Saunsar': 566, 'Loharda': 376, 'Dhamnod': 145, 'Gandhwani': 180, 'Kukshi': 362, 'Manawar': 400, 'Rajgarh': 521, 'Khirakiya': 336, 'Jhabua': 274, 'Petlawad': 499, 'Thandla': 626, 'Khandwa': 331, 'Pandhana': 479, 'Badwaha': 41, 'Bhikangaon': 79, 'Karhi': 309, 'Kasrawad': 314, 'Khargone': 333, 'Sanawad': 552, 'Segaon': 574, 'Ratlam': 537, 'Sailana': 547, 'Newasa': 461, 'Pathardi': 495, 'Sangamner': 553, 'Shevgaon': 582, 'Shrigonda': 588, 'Shrigonda(Gogargaon)': 589, 'Akot': 10, 'Barshi Takli': 54, 'Telhara': 623, 'Amarawati': 19, 'Chandur Railway': 119, 'Daryapur': 138, 'Dhamngaon-Railway': 144, 'Varud': 667, 'Fulmbri': 171, 'Khultabad': 337, 'Lasur Station': 371, 'Soygaon': 605, 'Vaijpur': 659, 'Gevrai': 187, 'Kille Dharur': 338, 'Majalgaon': 390, 'Vadvani': 657, 'Deoulgaon Raja': 142, 'Khamgaon': 328, 'Chimur': 125, 'Korpana': 350, 'Rajura': 525, 'Varora': 666, 'Shirpur': 586, 'Aheri': 6, 'Akhadabalapur': 8, 'Basmat': 57, 'Hingoli': 229, 'Jawala-Bajar': 270, 'Kalamnuri': 293, 'Bhokardan': 86, 'Ghansawangi': 189, 'Jalana': 254, 'Jalna(Badnapur)': 256, 'Mantha': 405, 'Partur': 490, 'Jalgaon': 255, 'Jamner': 266, 'Jamner(Neri)': 267, 'Yawal': 693, 'Bhiwapur': 84, 'Kalmeshwar': 297, 'Katol': 317, 'Mandhal': 402, 'Narkhed': 449, 'Parshiwani': 489, 'Savner': 572, 'Umared': 647, 'Bhokar': 85, 'Hadgaon': 211, 'Hadgaon(Tamsa)': 212, 'Himalyatnagar': 226, 'Kinwat': 340, 'Mahur': 387, 'Naigaon': 437, 'Shahada': 579, 'Gangakhed': 182, 'Jintur': 276, 'Manwat': 408, 'Parbhani': 485, 'Purna': 515, 'Selu': 575, 'Sonpeth': 603, 'Aatpadi': 0, 'Arvi': 33, 'Ashti': 34, 'Hinganghat': 228, 'Pulgaon': 513, 'Samudrapur': 551, 'Sindi': 595, 'Sindi(Selu)': 596, 'Wardha': 689, 'Babhulgaon': 37, 'Darwha': 137, 'Digras': 159, 'Ghatanji': 191, 'Kalamb': 292, 'Mahagaon': 386, 'Maregoan': 410, 'Pandhakawada': 478, 'Ralegaon': 526, 'Umarkhed': 648, 'Vani': 663, 'Yeotmal': 698, 'ZariZamini': 700, 'Mawkyrwat': 415, 'Nongstoin': 467, 'Bolangir(Patnagarh)': 106, 'Kasinagar': 313, 'Parlakhemundi': 488, 'Digapahandi': 158, 'Bhawanipatna': 77, 'Kesinga': 322, 'Koraput': 347, 'Koraput(Semilguda)': 348, 'Gunpur': 208, 'Rayagada': 540, 'Rayagada(Muniguda)': 541, 'Karaikal': 307, 'Madagadipet': 381, 'Thattanchavady': 629, 'Barnala': 53, 'Tapa(Tapa Mandi)': 621, 'Bhucho': 88, 'Maur': 414, 'Raman': 527, 'Rampura Phul': 530, 'Rampuraphul(Nabha Mandi)': 531, 'Sangat': 554, 'Faridkot': 168, 'Jaitu': 252, 'Kotkapura': 355, 'Abohar': 1, 'Fazilka': 170, 'Jagraon': 248, 'Bareta': 51, 'Bhikhi': 80, 'Boha': 105, 'Budalada': 110, 'Sardulgarh': 561, 'Bariwala': 52, 'Giddarbaha': 192, 'Malout': 394, 'Muktsar': 426, 'Ahmedgarh': 7, 'Dhuri': 157, 'Sangrur': 556, 'Sunam': 612, 'Beawar': 63, 'Bijay Nagar': 91, 'Kekri': 319, 'Vijay Nagar(Gulabpura)': 675, 'Alwar': 16, 'Khairthal': 324, 'Nagar': 435, 'Gangapur': 183, 'Anoopgarh': 28, 'Gajsinghpur': 177, 'Gharsana': 190, 'Jaitsar': 251, 'Kesarisinghpur': 321, 'Lalgarh Jatan': 369, 'Padampur': 471, 'Raisingh Nagar': 518, 'Rawla': 539, 'Sadulshahar': 546, 'Sri Karanpur': 606, 'Sri Vijayanagar': 607, 'Sriganganagar': 608, 'Bhadara': 68, 'Goluwala': 198, 'Hanumangarh': 217, 'Hanumangarh Town': 218, 'Hanumangarh(Urlivas)': 219, 'Nohar': 465, 'Pilli Banga': 501, 'Rawatsar': 538, 'Sangriya': 555, 'Suratgarh': 614, 'Bilara': 93, 'Mathania': 413, 'Merta City': 421, 'Rani': 534, 'Sumerpur': 611, 'Ariyalur Market': 31, 'Annur': 27, 'Pethappampatti': 498, 'Panruti': 481, 'Papparapatti': 483, 'Dindigul': 160, 'Natham': 458, 'Oddunchairum': 469, 'Vadamadurai': 655, 'Vedachandur': 669, 'Anthiyur': 29, 'Boothapadi': 107, 'Moolanur': 424, 'Sathyamangalam': 563, 'Uthangarai': 653, 'Thirumangalam': 632, 'Usilampatty': 652, 'Kuttulam': 367, 'Mailaduthurai': 389, 'Nagapattinam': 434, 'Sembanarkoil': 576, 'Sirkali': 600, 'Namakkal': 440, 'Paramakudi': 484, 'Gangavalli': 184, 'Karumanturai': 312, 'Kolathur': 344, 'Konganapuram': 346, 'Thalaivasal': 624, 'Vazhapadi': 668, 'Theni': 630, 'Sankarankovil': 557, 'Tirunelvali': 640, 'Kudavasal': 360, 'Thiruvarur': 636, 'Valangaiman': 660, 'Kovilpatti': 357, 'Gudiyatham': 205, 'Gingee': 193, 'Thirukovilur': 631, 'Tindivanam': 638, 'Vikkiravandi': 677, 'Villupuram': 678, 'Rajapalayam': 520, 'Virudhunagar': 682, 'Adilabad': 4, 'Asifabad': 35, 'Bhainsa': 72, 'Boath': 96, 'Chinnoar': 126, 'Ichoda': 242, 'Indravelly(Utnoor)': 245, 'Jainath': 249, 'Jainoor': 250, 'Kagaznagar': 290, 'Khanapur': 330, 'Kuber': 359, 'Laxettipet': 372, 'Mancharial': 401, 'Nirmal': 463, 'Sarangapur': 560, 'Choppadandi': 130, 'Dharmaram': 149, 'Gangadhara': 181, 'Gollapally': 197, 'Husnabad': 240, 'Jammikunta': 264, 'Karimnagar': 310, 'Kataram': 315, 'Mallial(Cheppial)': 393, 'Manthani': 406, 'Peddapalli': 496, 'Pudur': 512, 'Sircilla': 598, 'Sultanabad': 610, 'Vemulawada': 671, 'Bhadrachalam': 69, 'Burgampadu': 111, 'Charla': 121, 'Dammapet': 134, 'Enkoor': 167, 'Kallur': 296, 'Khammam': 329, 'Kothagudem': 354, 'Madhira': 382, 'Nelakondapally': 460, 'Sattupalli': 565, 'Wyra': 691, 'Yellandu': 694, 'Amangal': 18, 'Badepalli': 40, 'Makthal': 391, 'Narayanpet': 447, 'Shadnagar': 578, 'Dubbak': 165, 'Gajwel': 178, 'Jogipet': 279, 'Medak': 416, 'Narayankhed': 446, 'Narsapur': 454, 'Ramayampet': 528, 'Siddipet': 591, 'Togguta': 642, 'Zaheerabad': 699, 'Aler': 11, 'Chandur': 118, 'Chandur(Mungodu)': 120, 'Choutuppal': 132, 'Halia': 213, 'Nakrekal': 438, 'Venkateswarnagar': 672, 'Venkateswarnagar(Chintapalli)': 673, 'Armoor': 32, 'Banswada': 50, 'Bichkunda': 89, 'Bodhan': 102, 'Gandhari': 179, 'Kamareddy': 299, 'Kammarpally': 300, 'Madnoor': 384, 'Pitlam': 504, 'Yellareddy': 696, 'Marapally': 409, 'Medchal': 417, 'Vikarabad': 676, 'Cherial': 122, 'Dornakal': 164, 'Ghanpur': 188, 'Jangaon': 268, 'Kesamudram': 320, 'Kodakandal': 341, 'Mahabubabad': 385, 'Mulugu': 429, 'Narsampet': 452, 'Narsampet(Nekonda)': 453, 'Parkal': 487, 'Thorrur': 637, 'Warangal': 688, 'Wardhannapet': 690, 'Haathras': 210, 'Peddapuram': 497, 'Ponduru': 507, 'Nippani': 462, 'Rona': 543, 'Byadagi': 113, 'Devadurga': 143, 'Sira': 597, 'Dharni': 150, 'Tiwasa': 641, 'Kannad': 303, 'Sillod': 592, 'Chikali': 123, 'Malkapur': 392, 'Bhadrawati': 71, 'Chandrapur': 116, 'Dondaicha': 163, 'Navapur': 459, 'Mangrulpeer': 403, 'Goniana': 201, 'Pudupalayam': 511, 'Gopalpatti': 203, 'Tiruchengode': 639, 'Omalur': 470, 'Poonthottam': 508, 'Kathalapur': 316, 'Koratla': 349, 'Gadwal': 174, 'Gadwal(Lezza)': 175, 'Sadasivpet': 545, 'Bhiknoor': 81, 'Sadashivnagar': 544, 'Hodal': 234, 'K.R. Pet': 285, 'Sakri': 548, 'Hingoli(Kanegoan Naka)': 230, 'Partur(Vatur)': 491, 'Bodwad': 104, 'Nanded': 441, 'Mulshi': 428, 'Birmaharajpur': 95, 'Kapasan': 306, 'Kavunthapadi': 318, 'Thirupathur': 633, 'Wanaparthy Road': 686, 'Wanaparthy Road(Prbbair)': 687, 'Birkur': 94, 'Velpur': 670, 'Aligarh': 12, 'Khair': 323, 'Khambhat': 327, 'Deesa': 140, 'Lakhani': 368, 'Baruch(Vagara)': 55, 'Gogamba(Similiya)': 195, 'Talod(Harsol)': 620, 'Loharu': 377, 'Ballabhgarh': 47, 'Hissar': 233, 'Khanina': 332, 'Narnaul': 450, 'Shahapur': 580, 'Harsood': 222, 'Chandrapur(Ganjwad)': 117, 'Gondpimpri': 200, 'Nagpur(Hingna)': 436, 'Ramtek': 532, 'Nandurbar': 444, 'Bathinda': 59, 'Khajuwala': 325, 'Lunkaransar': 379, 'Nokha': 466, 'Ridmalsar': 542, 'Jodhpur(Grain)(Phalodi)': 278, 'Surajgarh': 613, 'Karamadai': 308, 'Thiruppur': 635, 'Namagiripettai': 439, 'Kamuthi': 301, 'Salem': 549, 'Thammampati': 625, 'Manamadurai': 398, 'Kumbakonam': 363, 'Papanasam': 482, 'Thiruppananthal': 634, 'Bodinayakkanur': 103, 'Vaniyambadi': 664, 'Sathur': 562, 'Gopalraopet': 204, 'Huzzurabad': 241, 'Medipally': 418, 'Kosikalan': 353, 'Porbandar': 509, 'Junagarh': 283, 'Pollachi': 506, 'Palani': 474, 'Bhongir': 87, 'Thara': 627, 'Thara(Shihori)': 628, 'Bharuch': 74, 'Jagalur': 246, 'Kustagi': 366, 'Madhugiri': 383, 'Akola': 9, 'Achalpur': 2, 'Anajngaon': 23, 'Paithan': 472, 'Nandura': 443, 'Shegaon': 581, 'Basmat(Kurunda)': 58, 'Pusad': 516, 'Malout (Kilianwali)': 395, 'Punchaipuliyampatti': 514, 'Kilvelur': 339}


def cotton(field):
	input_state = field['state']
	input_district = field['district']
	input_market = field['market']
	input_date = field['date'].split("-")

	#YYYY-MM-DD

	if int(input_date[2])<10:
		day=int(input_date[2][1:])

	else:
		day=int(input_date[2])


	if int(input_date[1])<10:
		month=int(input_date[1][1:])

	else:
		month=int(input_date[1])

	year=int(input_date[0])


	#df = pd.read_csv('/home/ubuntu/flaskproject/CottonData.csv')
	#dummy=df.drop('Price',axis='columns')

	"""dummy.loc[len(dummy.index)] = [day, month, year,input_state,input_district,input_market]

	le = preprocessing.LabelEncoder()
	X_2 = dummy.apply(le.fit_transform)

	enc = preprocessing.OneHotEncoder()
	enc.fit(X_2)

	onehotlabels = enc.transform(X_2).toarray()

	single_point=onehotlabels[len(onehotlabels)-1:]"""

	bin_days=[0]*31
	bin_months=[0]*12
	bin_years=[0]*6
	bin_states=[0]*14
	bin_districts=[0]*163
	bin_markets=[0]*702

	bin_days[day-1]=1
	bin_months[month-1]=1
	bin_years[year-2016]=1

	bin_states[state[input_state]]=1
	bin_districts[district[input_district]]=1
	bin_markets[market[input_market]]=1

	single_point=bin_days+bin_months+bin_years+bin_states+bin_districts+bin_markets

	single_point=np.array(single_point).reshape(1,928)



	pred_result = int(prophet_model.predict(single_point))

	return str(pred_result)

#################

app = Flask(__name__)

@app.route('/home')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/printpredict')
def printpredict():
    if request.method=='GET':
         return render_template('printpredict.html')
        

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
      if request.method == 'POST':
        post_state = request.form['state']
        post_district = request.form['district']
        post_market = request.form['market']
        post_date=request.form['date']
        
        all_post={'state': post_state,
                   'district' :post_district,
                    'market':post_market,
                       'date' :post_date,
                  
                  }
        value=cotton(all_post)
        all_post['price']=value
        return render_template('printpredict.html',post=all_post)
      else:
        return render_template('prediction.html')

if __name__ == "__main__":
    app.run(debug=True)


    
