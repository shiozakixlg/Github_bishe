#! /usr/bin/env python
# -*- coding: utf-8 -*- 

__author__ = 'gifrim'

# Example run:
# python twitter-topics-from-json-text-stream.py json-to-text-stream-syria.json.txt 15 15mins-topics-syria-stream.txt > details_clusters_15mins_topics_syria-stream.txt 
# python twitter-topics-from-json-text-stream.py data.txt 1440 2-19-test.txt

import codecs
from collections import Counter
from cmu import CMUTweetTagger
from datetime import datetime
import fastcluster
from itertools import cycle
import json
import nltk
import numpy as np
import re
#import requests
import os
import scipy.cluster.hierarchy as sch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import metrics
#from stemming.porter2 import stem
import chardet
import random
from bson.objectid import ObjectId
from mongoengine import *
import json
from mongoengine.queryset.visitor import Q
import string
import sys
import jieba
import time
reload(sys)
sys.setdefaultencoding('utf8')

connect('yuqing', alias='default', host='117.32.155.61', port=10005, username='yuqing', password='yuqing')

def load_stopwords():
	stop_words = nltk.corpus.stopwords.words('english')
	stop_words.extend([u'啊', u'阿', u'哎', u'哎呀', u'哎哟', u'唉', u'俺', u'俺们', u'按', u'按照', u'吧', u'吧哒', u'把', u'罢了', u'被', u'本', u'本着', u'比', u'比方', u'比如', u'鄙人', u'彼', u'彼此', u'边', u'别', u'别的', u'别说', u'并', u'并且', u'不比', u'不成', u'不单', u'不但', u'不独', u'不管', u'不光', u'不过', u'不仅', u'不拘', u'不论', u'不怕', u'不然', u'不如', u'不特', u'不惟', u'不问', u'不只', u'朝', u'朝着', u'趁', u'趁着', u'乘', u'冲', u'除', u'除此之外', u'除非', u'除了', u'此', u'此间', u'此外', u'从', u'从而', u'打', u'待', u'但', u'但是', u'当', u'当着', u'到', u'得', u'的', u'的话', u'等', u'等等', u'地', u'第', u'叮咚', u'对', u'对于', u'多', u'多少', u'而', u'而况', u'而且', u'而是', u'而外', u'而言', u'而已', u'尔后', u'反过来', u'反过来说', u'反之', u'非但', u'非徒', u'否则', u'嘎', u'嘎登', u'该', u'赶', u'个', u'各', u'各个', u'各位', u'各种', u'各自', u'给', u'根据', u'跟', u'故', u'故此', u'固然', u'关于', u'管', u'归', u'果然', u'果真', u'过', u'哈', u'哈哈', u'呵', u'和', u'何', u'何处', u'何况', u'何时', u'嘿', u'哼', u'哼唷', u'呼哧', u'乎', u'哗', u'还是', u'还有', u'换句话说', u'换言之', u'或', u'或是', u'或者', u'极了', u'及', u'及其', u'及至', u'即', u'即便', u'即或', u'即令', u'即若', u'即使', u'几', u'几时', u'己', u'既', u'既然', u'既是', u'继而', u'加之', u'假如', u'假若', u'假使', u'鉴于', u'将', u'较', u'较之', u'叫', u'接着', u'结果', u'借', u'紧接着', u'进而', u'尽', u'尽管', u'经', u'经过', u'就', u'就是', u'就是说', u'据', u'具体地说', u'具体说来', u'开始', u'开外', u'靠', u'咳', u'可', u'可见', u'可是', u'可以', u'况且', u'啦', u'来', u'来着', u'离', u'例如', u'哩', u'连', u'连同', u'两者', u'了', u'临', u'另', u'另外', u'另一方面', u'论', u'嘛', u'吗', u'慢说', u'漫说', u'冒', u'么', u'每', u'每当', u'们', u'莫若', u'某', u'某个', u'某些', u'拿', u'哪', u'哪边', u'哪儿', u'哪个', u'哪里', u'哪年', u'哪怕', u'哪天', u'哪些', u'哪样', u'那', u'那边', u'那儿', u'那个', u'那会儿', u'那里', u'那么', u'那么些', u'那么样', u'那时', u'那些', u'那样', u'乃', u'乃至', u'呢', u'能', u'你', u'你们', u'您', u'宁', u'宁可', u'宁肯', u'宁愿', u'哦', u'呕', u'啪达', u'旁人', u'呸', u'凭', u'凭借', u'其', u'其次', u'其二', u'其他', u'其它', u'其一', u'其余', u'其中', u'起', u'起见', u'岂但', u'恰恰相反', u'前后', u'前者', u'且', u'然而', u'然后', u'然则', u'让', u'人家', u'任', u'任何', u'任凭', u'如', u'如此', u'如果', u'如何', u'如其', u'如若', u'如上所述', u'若', u'若非', u'若是', u'啥', u'上下', u'尚且', u'设若', u'设使', u'甚而', u'甚么', u'甚至', u'省得', u'时候', u'什么', u'什么样', u'使得', u'是', u'是的', u'首先', u'谁', u'谁知', u'顺', u'顺着', u'似的', u'虽', u'虽然', u'虽说', u'虽则', u'随', u'随着', u'所', u'所以', u'他', u'他们', u'他人', u'它', u'它们', u'她', u'她们', u'倘', u'倘或', u'倘然', u'倘若', u'倘使', u'腾', u'替', u'通过', u'同', u'同时', u'哇', u'万一', u'往', u'望', u'为', u'为何', u'为了', u'为什么', u'为着', u'喂', u'嗡嗡', u'我', u'我们', u'呜', u'呜呼', u'乌乎', u'无论', u'无宁', u'毋宁', u'嘻', u'吓', u'相对而言', u'像', u'向', u'向着', u'嘘', u'呀', u'焉', u'沿', u'沿着', u'要', u'要不', u'要不然', u'要不是', u'要么', u'要是', u'也', u'也罢', u'也好', u'一', u'一般', u'一旦', u'一方面', u'一来', u'一切', u'一样', u'一则', u'依', u'依照', u'矣', u'以', u'以便', u'以及', u'以免', u'以至', u'以至于', u'以致', u'抑或', u'因', u'因此', u'因而', u'因为', u'哟', u'用', u'由', u'由此可见', u'由于', u'有', u'有的', u'有关', u'有些', u'又', u'于', u'于是', u'于是乎', u'与', u'与此同时', u'与否', u'与其', u'越是', u'云云', u'哉', u'再说', u'再者', u'在', u'在下', u'咱', u'咱们', u'则', u'怎', u'怎么', u'怎么办', u'怎么样', u'怎样', u'咋', u'照', u'照着', u'者', u'这', u'这边', u'这儿', u'这个', u'这会儿', u'这就是说', u'这里', u'这么', u'这么点儿', u'这么些', u'这么样', u'这时', u'这些', u'这样', u'正如', u'吱', u'之', u'之类', u'之所以', u'之一', u'只是', u'只限', u'只要', u'只有', u'至', u'至于', u'诸位', u'着', u'着呢', u'自', u'自从', u'自个儿', u'自各儿', u'自己', u'自家', u'自身', u'综上所述', u'总的来看', u'总的来说', u'总的说来', u'总而言之', u'总之', u'纵', u'纵令', u'纵然', u'纵使', u'遵照', u'作为', u'兮', u'呃', u'呗', u'咚', u'咦', u'喏', u'啐', u'喔唷', u'嗬', u'嗯', u'嗳', u'啊哈', u'啊呀', u'啊哟', u'挨次', u'挨个', u'挨家挨户', u'挨门挨户', u'挨门逐户', u'挨着', u'按理', u'按期', u'按时', u'按说', u'暗地里', u'暗中', u'暗自', u'昂然', u'八成', u'白白', u'半', u'梆', u'保管', u'保险', u'饱', u'背地里', u'背靠背', u'倍感', u'倍加', u'本人', u'本身', u'甭', u'比起', u'比如说', u'比照', u'毕竟', u'必', u'必定', u'必将', u'必须', u'便', u'别人', u'并非', u'并肩', u'并没', u'并没有', u'并排', u'并无', u'勃然', u'不', u'不必', u'不常', u'不大', u'不得', u'不得不', u'不得了', u'不得已', u'不迭', u'不定', u'不对', u'不妨', u'不管怎样', u'不会', u'不仅仅', u'不仅仅是', u'不经意', u'不可开交', u'不可抗拒', u'不力', u'不了', u'不料', u'不满', u'不免', u'不能不', u'不起', u'不巧', u'不然的话', u'不日', u'不少', u'不胜', u'不时', u'不是', u'不同', u'不能', u'不要', u'不外', u'不外乎', u'不下', u'不限', u'不消', u'不已', u'不亦乐乎', u'不由得', u'不再', u'不择手段', u'不怎么', u'不曾', u'不知不觉', u'不止', u'不止一次', u'不至于', u'才', u'才能', u'策略地', u'差不多', u'差一点', u'常', u'常常', u'常言道', u'常言说', u'常言说得好', u'长此下去', u'长话短说', u'长期以来', u'长线', u'敞开儿', u'彻夜', u'陈年', u'趁便', u'趁机', u'趁热', u'趁势', u'趁早', u'成年', u'成年累月', u'成心', u'乘机', u'乘胜', u'乘势', u'乘隙', u'乘虚', u'诚然', u'迟早', u'充分', u'充其极', u'充其量', u'抽冷子', u'臭', u'初', u'出', u'出来', u'出去', u'除此', u'除此而外', u'除此以外', u'除开', u'除去', u'除却', u'除外', u'处处', u'川流不息', u'传', u'传说', u'传闻', u'串行', u'纯', u'纯粹', u'此后', u'此中', u'次第', u'匆匆', u'从不', u'从此', u'从此以后', u'从古到今', u'从古至今', u'从今以后', u'从宽', u'从来', u'从轻', u'从速', u'从头', u'从未', u'从无到有', u'从小', u'从新', u'从严', u'从优', u'从早到晚', u'从中', u'从重', u'凑巧', u'粗', u'存心', u'达旦', u'打从', u'打开天窗说亮话', u'大', u'大不了', u'大大', u'大抵', u'大都', u'大多', u'大凡', u'大概', u'大家', u'大举', u'大略', u'大面儿上', u'大事', u'大体', u'大体上', u'大约', u'大张旗鼓', u'大致', u'呆呆地', u'带', u'殆', u'待到', u'单', u'单纯', u'单单', u'但愿', u'弹指之间', u'当场', u'当儿', u'当即', u'当口儿', u'当然', u'当庭', u'当头', u'当下', u'当真', u'当中', u'倒不如', u'倒不如说', u'倒是', u'到处', u'到底', u'到了儿', u'到目前为止', u'到头', u'到头来', u'得起', u'得天独厚', u'的确', u'等到', u'叮当', u'顶多', u'定', u'动不动', u'动辄', u'陡然', u'都', u'独', u'独自', u'断然', u'顿时', u'多次', u'多多', u'多多少少', u'多多益善', u'多亏', u'多年来', u'多年前', u'而后', u'而论', u'而又', u'尔等', u'二话不说', u'二话没说', u'反倒', u'反倒是', u'反而', u'反手', u'反之亦然', u'反之则', u'方', u'方才', u'方能', u'放量', u'非常', u'非得', u'分期', u'分期分批', u'分头', u'奋勇', u'愤然', u'风雨无阻', u'逢', u'弗', u'甫', u'嘎嘎', u'该当', u'概', u'赶快', u'赶早不赶晚', u'敢', u'敢情', u'敢于', u'刚', u'刚才', u'刚好', u'刚巧', u'高低', u'格外', u'隔日', u'隔夜', u'个人', u'各式', u'更', u'更加', u'更进一步', u'更为', u'公然', u'共', u'共总', u'够瞧的', u'姑且', u'古来', u'故而', u'故意', u'固', u'怪', u'怪不得', u'惯常', u'光', u'光是', u'归根到底', u'归根结底', u'过于', u'毫不', u'毫无', u'毫无保留地', u'毫无例外', u'好在', u'何必', u'何尝', u'何妨', u'何苦', u'何乐而不为', u'何须', u'何止', u'很', u'很多', u'很少', u'轰然', u'后来', u'呼啦', u'忽地', u'忽然', u'互', u'互相', u'哗啦', u'话说', u'还', u'恍然', u'会', u'豁然', u'活', u'伙同', u'或多或少', u'或许', u'基本', u'基本上', u'基于', u'极', u'极大', u'极度', u'极端', u'极力', u'极其', u'极为', u'急匆匆', u'即将', u'即刻', u'即是说', u'几度', u'几番', u'几乎', u'几经', u'既…又', u'继之', u'加上', u'加以', u'间或', u'简而言之', u'简言之', u'简直', u'见', u'将才', u'将近', u'将要', u'交口', u'较比', u'较为', u'接连不断', u'接下来', u'皆可', u'截然', u'截至', u'藉以', u'借此', u'借以', u'届时', u'仅', u'仅仅', u'谨', u'进来', u'进去', u'近', u'近几年来', u'近来', u'近年来', u'尽管如此', u'尽可能', u'尽快', u'尽量', u'尽然', u'尽如人意', u'尽心竭力', u'尽心尽力', u'尽早', u'精光', u'经常', u'竟', u'竟然', u'究竟', u'就此', u'就地', u'就算', u'居然', u'局外', u'举凡', u'据称', u'据此', u'据实', u'据说', u'据我所知', u'据悉', u'具体来说', u'决不', u'决非', u'绝', u'绝不', u'绝顶', u'绝对', u'绝非', u'均', u'喀', u'看', u'看来', u'看起来', u'看上去', u'看样子', u'可好', u'可能', u'恐怕', u'快', u'快要', u'来不及', u'来得及', u'来讲', u'来看', u'拦腰', u'牢牢', u'老', u'老大', u'老老实实', u'老是', u'累次', u'累年', u'理当', u'理该', u'理应', u'历', u'立', u'立地', u'立刻', u'立马', u'立时', u'联袂', u'连连', u'连日', u'连日来', u'连声', u'连袂', u'临到', u'另方面', u'另行', u'另一个', u'路经', u'屡', u'屡次', u'屡次三番', u'屡屡', u'缕缕', u'率尔', u'率然', u'略', u'略加', u'略微', u'略为', u'论说', u'马上', u'蛮', u'满', u'没', u'没有', u'每逢', u'每每', u'每时每刻', u'猛然', u'猛然间', u'莫', u'莫不', u'莫非', u'莫如', u'默默地', u'默然', u'呐', u'那末', u'奈', u'难道', u'难得', u'难怪', u'难说', u'内', u'年复一年', u'凝神', u'偶而', u'偶尔', u'怕', u'砰', u'碰巧', u'譬如', u'偏偏', u'乒', u'平素', u'颇', u'迫于', u'扑通', u'其后', u'其实', u'奇', u'齐', u'起初', u'起来', u'起首', u'起头', u'起先', u'岂', u'岂非', u'岂止', u'迄', u'恰逢', u'恰好', u'恰恰', u'恰巧', u'恰如', u'恰似', u'千', u'万', u'千万', u'千万千万', u'切', u'切不可', u'切莫', u'切切', u'切勿', u'窃', u'亲口', u'亲身', u'亲手', u'亲眼', u'亲自', u'顷', u'顷刻', u'顷刻间', u'顷刻之间', u'请勿', u'穷年累月', u'取道', u'去', u'权时', u'全都', u'全力', u'全年', u'全然', u'全身心', u'然', u'人人', u'仍', u'仍旧', u'仍然', u'日复一日', u'日见', u'日渐', u'日益', u'日臻', u'如常', u'如此等等', u'如次', u'如今', u'如期', u'如前所述', u'如上', u'如下', u'汝', u'三番两次', u'三番五次', u'三天两头', u'瑟瑟', u'沙沙', u'上', u'上来', u'上去', u'一.', u'一一', u'一下', u'一个', u'一些', u'一何', u'一则通过', u'一天', u'一定', u'一时', u'一次', u'一片', u'一番', u'一直', u'一致', u'一起', u'一转眼', u'一边', u'一面', u'上升', u'上述', u'上面', u'下', u'下列', u'下去', u'下来', u'下面', u'不一', u'不久', u'不变', u'不可', u'不够', u'不尽', u'不尽然', u'不敢', u'不断', u'不若', u'不足', u'与其说', u'专门', u'且不说', u'且说', u'严格', u'严重', u'个别', u'中小', u'中间', u'丰富', u'为主', u'为什麽', u'为止', u'为此', u'主张', u'主要', u'举行', u'乃至于', u'之前', u'之后', u'之後', u'也就是说', u'也是', u'了解', u'争取', u'二来', u'云尔', u'些', u'亦', u'产生', u'人', u'人们', u'什麽', u'今', u'今后', u'今天', u'今年', u'今後', u'介于', u'从事', u'他是', u'他的', u'代替', u'以上', u'以下', u'以为', u'以前', u'以后', u'以外', u'以後', u'以故', u'以期', u'以来', u'任务', u'企图', u'伟大', u'似乎', u'但凡', u'何以', u'余外', u'你是', u'你的', u'使', u'使用', u'依据', u'依靠', u'便于', u'促进', u'保持', u'做到', u'傥然', u'儿', u'允许', u'元／吨', u'先不先', u'先后', u'先後', u'先生', u'全体', u'全部', u'全面', u'共同', u'具体', u'具有', u'兼之', u'再', u'再其次', u'再则', u'再有', u'再次', u'再者说', u'决定', u'准备', u'凡', u'凡是', u'出于', u'出现', u'分别', u'则甚', u'别处', u'别是', u'别管', u'前此', u'前进', u'前面', u'加入', u'加强', u'十分', u'即如', u'却', u'却不', u'原来', u'又及', u'及时', u'双方', u'反应', u'反映', u'取得', u'受到', u'变成', u'另悉', u'只', u'只当', u'只怕', u'只消', u'叫做', u'召开', u'各人', u'各地', u'各级', u'合理', u'同一', u'同样', u'后', u'后者', u'后面', u'向使', u'周围', u'呵呵', u'咧', u'唯有', u'啷当', u'喽', u'嗡', u'嘿嘿', u'因了', u'因着', u'在于', u'坚决', u'坚持', u'处在', u'处理', u'复杂', u'多么', u'多数', u'大力', u'大多数', u'大批', u'大量', u'失去', u'她是', u'她的', u'好', u'好的', u'好象', u'如同', u'如是', u'始而', u'存在', u'孰料', u'孰知', u'它们的', u'它是', u'它的', u'安全', u'完全', u'完成', u'实现', u'实际', u'宣布', u'容易', u'密切', u'对应', u'对待', u'对方', u'对比', u'小', u'少数', u'尔', u'尔尔', u'尤其', u'就是了', u'就要', u'属于', u'左右', u'巨大', u'巩固', u'已', u'已矣', u'已经', u'巴', u'巴巴', u'帮助', u'并不', u'并不是', u'广大', u'广泛', u'应当', u'应用', u'应该', u'庶乎', u'庶几', u'开展', u'引起', u'强烈', u'强调', u'归齐', u'当前', u'当地', u'当时', u'形成', u'彻底', u'彼时', u'往往', u'後来', u'後面', u'得了', u'得出', u'得到', u'心里', u'必然', u'必要', u'怎奈', u'怎麽', u'总是', u'总结', u'您们', u'您是', u'惟其', u'意思', u'愿意', u'成为', u'我是', u'我的', u'或则', u'或曰', u'战斗', u'所在', u'所幸', u'所有', u'所谓', u'扩大', u'掌握', u'接著', u'数/', u'整个', u'方便', u'方面', u'无', u'无法', u'既往', u'明显', u'明确', u'是不是', u'是以', u'是否', u'显然', u'显著', u'普通', u'普遍', u'曾', u'曾经', u'替代', u'最', u'最后', u'最大', u'最好', u'最後', u'最近', u'最高', u'有利', u'有力', u'有及', u'有所', u'有效', u'有时', u'有点', u'有的是', u'有着', u'有著', u'末##末', u'本地', u'来自', u'来说', u'构成', u'某某', u'根本', u'欢迎', u'欤', u'正值', u'正在', u'正巧', u'正常', u'正是', u'此地', u'此处', u'此时', u'此次', u'每个', u'每天', u'每年', u'比及', u'比较', u'没奈何', u'注意', u'深入', u'清楚', u'满足', u'然後', u'特别是', u'特殊', u'特点', u'犹且', u'犹自', u'现代', u'现在', u'甚且', u'甚或', u'甚至于', u'用来', u'由是', u'由此', u'目前', u'直到', u'直接', u'相似', u'相信', u'相反', u'相同', u'相对', u'相应', u'相当', u'相等', u'看出', u'看到', u'看看', u'看见', u'真是', u'真正', u'眨眼', u'矣乎', u'矣哉', u'知道', u'确定', u'种', u'积极', u'移动', u'突出', u'突然', u'立即', u'竟而', u'第二', u'类如', u'练习', u'组成', u'结合', u'继后', u'继续', u'维持', u'考虑', u'联系', u'能否', u'能够', u'自后', u'自打', u'至今', u'至若', u'致', u'般的', u'良好', u'若夫', u'若果', u'范围', u'莫不然', u'获得', u'行为', u'行动', u'表明', u'表示', u'要求', u'规定', u'觉得', u'譬喻', u'认为', u'认真', u'认识', u'许多', u'设或', u'诚如', u'说明', u'说来', u'说说', u'诸', u'诸如', u'谁人', u'谁料', u'贼死', u'赖以', u'距', u'转动', u'转变', u'转贴', u'达到', u'迅速', u'过去', u'过来', u'运用', u'还要', u'这一来', u'这次', u'这点', u'这种', u'这般', u'这麽', u'进入', u'进步', u'进行', u'适应', u'适当', u'适用', u'逐步', u'逐渐', u'通常', u'造成', u'遇到', u'遭到', u'遵循', u'避免', u'那般', u'那麽', u'部分', u'采取', u'里面', u'重大', u'重新', u'重要', u'针对', u'问题', u'防止', u'附近', u'限制', u'随后', u'随时', u'随著', u'难道说', u'集中', u'需要', u'非特', u'非独', u'高兴', u'若果', u'...',u'全文',u'##',u'展开',u'还有',u'就是',u'秒拍视频',u'秒拍'])
	stop_words = set(stop_words)
	return stop_words

#将文本格式正常化为utf8
def normalize_text(text):
	try:
		text = text.encode('utf-8')
	except: pass
	text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', text)
	text = re.sub('@[^\s]+','', text)
	text = re.sub('#([^\s]+)', '', text)
	text = re.sub('[:;>?<=*+()/,\-#!$%\{˜|\}\[^_\\@\]1234567890’‘]',' ', text)
	text = re.sub('[\d]','', text)
	text = text.replace(".", '')
	text = text.replace("'", ' ')
	text = text.replace("\"", ' ')
	#text = text.replace("-", " ")
	#normalize some utf8 encoding
	text = text.replace("\x9d",' ').replace("\x8c",' ')
	text = text.replace("\xa0",' ')
	text = text.replace("\x9d\x92", ' ').replace("\x9a\xaa\xf0\x9f\x94\xb5", ' ').replace("\xf0\x9f\x91\x8d\x87\xba\xf0\x9f\x87\xb8", ' ').replace("\x9f",' ').replace("\x91\x8d",' ')
	text = text.replace("\xf0\x9f\x87\xba\xf0\x9f\x87\xb8",' ').replace("\xf0",' ').replace('\xf0x9f','').replace("\x9f\x91\x8d",' ').replace("\x87\xba\x87\xb8",' ')	
	text = text.replace("\xe2\x80\x94",' ').replace("\x9d\xa4",' ').replace("\x96\x91",' ').replace("\xe1\x91\xac\xc9\x8c\xce\x90\xc8\xbb\xef\xbb\x89\xd4\xbc\xef\xbb\x89\xc5\xa0\xc5\xa0\xc2\xb8",' ')
	text = text.replace("\xe2\x80\x99s", " ").replace("\xe2\x80\x98", ' ').replace("\xe2\x80\x99", ' ').replace("\xe2\x80\x9c", " ").replace("\xe2\x80\x9d", " ")
	text = text.replace("\xe2\x82\xac", " ").replace("\xc2\xa3", " ").replace("\xc2\xa0", " ").replace("\xc2\xab", " ").replace("\xf0\x9f\x94\xb4", " ").replace("\xf0\x9f\x87\xba\xf0\x9f\x87\xb8\xf0\x9f", "")
	return text



#分出一个个特征词
def nltk_tokenize(text):
	tokens = []
	pos_tokens = []
	entities = []
	features = []
	

	try:

			#分词,去空格
			# tokens = text.split() #英语
			tokens_cut = jieba.cut(text)

			for word in tokens_cut:
				#如果不是停止词并且长度大于1,在特征中加上这个单词
				if word not in stop_words and len(word) > 1:		
						#features.append(word + "." + postag)
					features.append(word)	

			for word in tokens_cut:
				tokens.append(word)
			# print 'feature here ', features
	except: pass
	return [tokens, pos_tokens, entities, features]				

#传入的text为未加工的生tweet文本	
def process_json_tweet(text, fout, debug):
	features = []

	#如果文本为空，返回空列表
	if len(text.strip()) == 0:
		return []
	#将文本变为utf8格式，并去除某些符号
	# print 'text raw' , text
	# text = normalize_text(text)
	# print 'text here:' , text
	#print text
	#nltk pre-processing: tokenize and pos-tag, try to extract entities
	try:
		[tokens, pos_tokens, entities, features] = nltk_tokenize(text)
	except: 
		print "nltk tokenize+pos pb!"
	if debug:	
		try:
			fout.write("\n--------------------clean text--------------------\n")
			fout.write(text.decode('utf-8'))
			#fout.write(text)
			fout.write("\n--------------------tokens--------------------\n")
			fout.write(str(tokens))
	#		fout.write("\n--------------------cleaned tokens--------------------\n")
	#		fout.write(str(clean_tokens))
			fout.write("\n--------------------pos tokens--------------------\n")
			fout.write(str(pos_tokens))
			fout.write("\n--------------------entities--------------------\n")
			for ent in entities:
				fout.write("\n" + str(ent).decode('utf-8'))
			fout.write("\n--------------------features--------------------\n")
			fout.write(str(features))
			fout.write("\n\n")
		except:
			#print "couldn't print text"
			pass
	return features

'''Prepare features, where doc has terms separated by comma'''
def custom_tokenize_text(text):
	REGEX = re.compile(r",\s*")
	tokens = []
	for tok in REGEX.split(text):
		#if "@" not in tok and "#" not in tok:
		if "@" not in tok:
			#tokens.append(stem(tok.strip().lower()))
			tokens.append(tok.strip().lower())
	return tokens		

#判断是否为僵尸博
def spam_tweet(text):
	if 'Jordan Bahrain Morocco Syria Qatar Oman Iraq Egypt United States' in text:
		return True
		
	if 'Some of you on my facebook are asking if it\'s me' in text:
		return True
		
	if '@kylieminogue please Kylie Follow Me, please' in text:
		return True
	
	if 'follow me please' in text:
		return True
	
	if 'please follow me' in text:
		return True		
			
	return False	
        
'''start main'''    
if __name__ == "__main__":
	# print '11111'

    #这里要改成丛数据库获取，并以时间排序,注意生成时间戳
	file_timeordered_tweets = codecs.open(sys.argv[1], 'r', 'utf-8') #打开数据集

	# print chardet.detect(file_timeordered_tweets)
    ##设定时间窗口,这里可以直接指定时间窗口为一天的大小
	time_window_mins = float(sys.argv[2]) 
	#file_timeordered_news = codecs.open(sys.argv[3], 'r', 'utf-8')
	fout = codecs.open(sys.argv[3], 'w', 'utf-8') #指定输出文件
	# print '2222222'



	debug=0
	stop_words = load_stopwords() #停用词集合 要改成中文
	# print chardet.detect(list(stop_words)[2])
					
	#read tweets in time order and window them				
	tweet_unixtime_old = -1 #时间戳
	#fout.write("time window size in mins: " + str(time_window_mins))

	tid_to_raw_tweet = {} #id与推文的对应字典
	window_corpus = [] #这个时间窗口的经过重构的推文列表
	tid_to_urls_window_corpus = {} #推特id与包含的媒体链接对应
	tids_window_corpus = [] #这个时间窗口的推特的id
	dfVocTimeWindows = {}
	t = 0
	ntweets = 0

	#取得每条推特进行处理
	for line in file_timeordered_tweets: 

		# print chardet.detect(line)
		# [tweet_unixtime, tweet_gmttime, tweet_id, text, hashtags, users, urls, media_urls, nfollowers, nfriends] = eval(line)
		[tweet_unixtime, tweet_gmttime, tweet_id, text, hashtags, users, urls, media_urls, nfollowers, nfriends] = eval(line)
		#判断是否为僵尸发的推特，是则处理下一条
		text = text.decode('unicode_escape').encode('utf-8')

		users[0] = users[0].decode('unicode_escape').encode('utf-8')
		if spam_tweet(text): 
			continue

		#如果时间戳过时，则更新时间戳。 这里因数据集并不密集，可以在之后考虑以每天0点作为阈值下限。
		if tweet_unixtime_old == -1:
			tweet_unixtime_old = tweet_unixtime

#  		#while this condition holds we are within the given size time window
		# print '21212121',int(tweet_unixtime) - int(tweet_unixtime_old)
		# print tweet_unixtime


		#在实验中以一天试验
		if (int(tweet_unixtime) - int(tweet_unixtime_old)) < time_window_mins * 60:
			# print time_window_mins * 60
			# print '333333333'
			ntweets += 1
				
			features = process_json_tweet(text, fout, debug) #传回来的是一个个word 或者说term
			# print 'here: ', features
			tweet_bag = ""
			#重构推文，@用户，话题，分词
			try:
				for user in set(users):
					tweet_bag += "@" + user.decode('utf-8').lower() + ","
				for tag in set(hashtags):
					if tag.decode('utf-8').lower() not in stop_words: 
						tweet_bag += "#" + tag.decode('utf-8').lower() + ","
				for feature in features:
					# print 'single feature', feature
					# if feature not in stop_words:
						tweet_bag += feature + ","
			except:
				#print "tweet_bag error!", tweet_bag, len(tweet_bag.split(","))
				pass

			#print tweet_bag.decode('utf-8')
			# print 'featuresssssss', features 

			if len(users) < 3 and len(hashtags) < 3 and len(features) > 3 and len(tweet_bag.split(",")) > 4 and not str(features).upper() == str(features):
				# print tweet_bag


				#tweet_bag的形式为@xxx,#xxx,xxx,xxx,xxx,xxx....
				tweet_bag = tweet_bag[:-1] #作用是去掉最后的逗号
				window_corpus.append(tweet_bag) #窗口语料集
				tids_window_corpus.append(tweet_id) #这个语料集里包含的id
				tid_to_urls_window_corpus[tweet_id] = media_urls #每个id对应推文包含的链接
				tid_to_raw_tweet[tweet_id] = text #id对应的推文

				# print tweet_bag

				# raw_input()
				
		else:
			if 1:
				# print window_corpus
				# print '44444444'
				print '1111'
				dtime = datetime.fromtimestamp(int(tweet_unixtime_old)).strftime("%d-%m-%Y %H:%M") #这个时间窗口的起始时间
				print "\nWindow Starts GMT Time:", dtime, "\n"
				tweet_unixtime_old = tweet_unixtime	#将最后一条推文的时间保存为下次的时间下限

				#每次到一个时间窗口t+1
				t += 1

				try:
					# texts=["dog cat fish","dog cat cat","fish bird", 'bird']
					# cv = CountVectorizer()
					# cv_fit=cv.fit_transform(texts)
					# print(cv.get_feature_names())
					# print(cv_fit.toarray())
					# #['bird', 'cat', 'dog', 'fish']
					# #[[0 1 1 1]
					# # [0 2 1 0]
					# # [1 0 0 1]
					# # [1 0 0 0]]
					# print(cv_fit.toarray().sum(axis=0))
					# #[2 3 2 2]
					
					#CountVectorizer提取tf都做了这些：去音调、转小写、去停顿词、在word（而不是character，也可自己选择参数）
					#基础上提取所有ngram_range范围内的特征，同时删去满足“max_df, min_df,max_features”的特征的tf。当然，也可以选择tf为binary。



					#该类将文本中的词语转化为词频矩阵，行为文本，列为term
					vectorizer = CountVectorizer(tokenizer=custom_tokenize_text, binary=True, min_df=max(int(len(window_corpus)*0.0025), 10), ngram_range=(1,1))
					#将这个语料集转化为词频矩阵，a[i][j]表示词j在第i类文本下的词频

	 				X = vectorizer.fit_transform(window_corpus)

	 				map_index_after_cleaning = {}
	 				#shape[n]表示矩阵第n+1维长度,np.zeros(n,m)生成n行m列第零矩阵,行为第一维
	 			# 	print X.toarray()
	 			# 	print X.shape[0]
					# print X.shape[1]
					# raw_input()
	 				Xclean = np.zeros((1, X.shape[1])) #一行，x.shape[1]列的零矩阵,shape[1]表示term的个数
	 				for i in range(0, X.shape[0]): #x.shape[0]表示有多少条推文
	 					#keep sample with size at least 5
	 					if X[i].sum() > 4: #一条推特中应包含5个以上的term
	 						#vstack 矩阵垂直合并 toarray转换成数组，这里是不断的将满足词语出现大于4的文本加入矩阵，矩阵第一行为零向量
	 						Xclean = np.vstack([Xclean, X[i].toarray()]) 
	 						#建立索引，干净矩阵第n行对应推特为i
	 						map_index_after_cleaning[Xclean.shape[0] - 2] = i #建立一个索引

							
	 				Xclean = Xclean[1:,] #去掉第一个零向量，因原来第一行为零向量
					
					#这个时间窗口进来的总推特数
					print "total tweets in window:", ntweets
					#print "len(window_corpus):", len(window_corpus)
					#矩阵的规格
					print "X.shape:", X.shape
					#去掉不符合文集后矩阵的规格
	 				print "Xclean.shape:", Xclean.shape

					#X变为简化后的矩阵
					X = Xclean
					#matrix为转换成矩阵，astype将数据转换为astype的类型
					Xdense = np.matrix(X).astype('float') #稠密矩阵
					#数据集的标准化？让矩阵数据更加约束，偏差较小
					X_scaled = preprocessing.scale(Xdense)
					#正则化，正则化的过程是将每个样本缩放到单位范数
					X_normalized = preprocessing.normalize(X_scaled, norm='l2')
					#transpose X to get features on the rows
					#Xt = X_scaled.T
	# 				#print "Xt.shape:", Xt.shape
					#获得term的名字
	 				vocX = vectorizer.get_feature_names()
	 				#print "Vocabulary (tweets):", vocX
	 				#sys.exit()
	 				
	 				boost_entity = {}
	 				#pos_tokens格式[[('example', 'N', 0.979), ('tweet', 'V', 0.7763), ('1', '$', 0.9916)], [('example', 'N', 0.979), ('tweet', 'V', 0.7713), ('2', '$', 0.5832)]]
	 				pos_tokens = CMUTweetTagger.runtagger_parse([term.upper() for term in vocX])
	 				#print "detect entities", pos_tokens
	 				#为每个term赋值一个权重
	 				for l in pos_tokens:
	 					term =''
	 					for gr in range(0, len(l)):
	 						#将每个单词合起来 用空格隔开
	 						term += l[gr][0].lower() + " "
	 					# print term
	 					# raw_input()
	  					if "^" in str(l):
	 						boost_entity[term.strip()] = 2.5
	 					else: 	 		
	 				 		boost_entity[term.strip()] = 1.0
		

					#接下来一段为计算term的df-idf的值
					#sum(axis=0)矩阵按列相加，统计一个词出现在这个时间窗口的出现频率
					dfX = X.sum(axis=0)

	 				dfVoc = {}
	 				wdfVoc = {}
	 				boosted_wdfVoc = {}	
	 				#vocX为term的列表，即关键字列表
	 				keys = vocX
	 				#一个词出现次数列表
	 				vals = dfX
	 				#k为单词名，v为出现次数，这里将term与出现次数对应
	 				for k,v in zip(keys, vals):
	 					dfVoc[k] = v
	 				#下面为计算对于一个term的df-idf值
	 				for k in dfVoc: 
	 					try:
	 						#定义为字典
	 						dfVocTimeWindows[k] += dfVoc[k]
	 						avgdfVoc = (dfVocTimeWindows[k] - dfVoc[k])/(t - 1)
	 
						except:
	 						dfVocTimeWindows[k] = dfVoc[k]
	 						avgdfVoc = 0
	 					#log为取对数，这里为df-idf公式
	 					wdfVoc[k] = (dfVoc[k] + 1) / (np.log(avgdfVoc + 1) + 1)
	 					#窗口增量相关
						try:
							boosted_wdfVoc[k] = wdfVoc[k] * boost_entity[k]
						except: 
							boosted_wdfVoc[k] = wdfVoc[k]

					print "sorted wdfVoc*boost_entity:"
					#输出频次与键值对，从高权值往下
					# print sorted(boosted_wdfVoc.keys().decode('utf8','ignore'))
					a = sorted( ((v,k) for k,v in boosted_wdfVoc.iteritems()), reverse=True)
					b = []
					for i in a:
						c = []
						# print i[1]
						# print i[0]
						# raw_input()
						c.append(i[1])
						c.append(i[0])
						if i[1] not in stop_words:
							b.append(c)
						else:
							pass
					print json.dumps(b,encoding="UTF-8", ensure_ascii=False)
					# print sorted( ((v,k) for k,v in boosted_wdfVoc.iteritems()), reverse=True)
	 							


					#计算每对推特距离，以cos作为标准
					distMatrix = pairwise_distances(X_normalized, metric='cosine')





					#利用fastcluster库进行分层聚类，hierarchical clustering
	 				print "fastcluster, average, cosine"
	 				L = fastcluster.linkage(distMatrix, method='average')


					#切割树图的阈值
					dt = 0.5
					print "hclust cut threshold:", dt
					#从分层聚类中形成flat clusters，每个簇距离不超过dt*distMatrix.max()
					indL = sch.fcluster(L, dt*distMatrix.max(), 'distance')
					#
					freqTwCl = Counter(indL)
					#聚簇数量
					print "n_clusters:", len(freqTwCl)
					print(freqTwCl)
					
					npindL = np.array(indL)

					print npindL#1
	#				
					#属于同一个话题的推特数的阈值
					#这块功能为计算各聚簇的score并排序显示，显示前50个，先过滤一部分不然太多
					freq_th = max(10, int(X.shape[0]*0.0025))
					cluster_score = {}
		 			for clfreq in freqTwCl.most_common(50):
		 				cl = clfreq[0]
		 				print cl #1
	 					freq = clfreq[1]

	 					print freq#1
	 					cluster_score[cl] = 0
	 					if freq >= freq_th:
	 	 					#print "\n(cluster, freq):", clfreq
		 					clidx = (npindL == cl).nonzero()[0].tolist()

		 					print clidx#1


		 					raw_input()#1
							cluster_centroid = X[clidx].sum(axis=0)
							#print "centroid_array:", cluster_centroid
							try:
								#orig_tweet = window_corpus[map_index_after_cleaning[i]].decode("utf-8")
								cluster_tweet = vectorizer.inverse_transform(cluster_centroid)
								#print orig_tweet, cluster_tweet, urls_window_corpus[map_index_after_cleaning[i]]
								#print orig_tweet
								#print "centroid_tweet:", cluster_tweet
								for term in np.nditer(cluster_tweet):
									#print "term:", term#, wdfVoc[term]
									try:
										cluster_score[cl] = max(cluster_score[cl], boosted_wdfVoc[str(term).strip()])
										#cluster_score[cl] += wdfVoc[str(term).strip()] * boost_entity[str(term)] #* boost_term_in_article[str(term)]
										#cluster_score[cl] = max(cluster_score[cl], wdfVoc[str(term).strip()] * boost_term_in_article[str(term)])
										#cluster_score[cl] = max(cluster_score[cl], wdfVoc[str(term).strip()] * boost_entity[str(term)])	
										#cluster_score[cl] = max(cluster_score[cl], wdfVoc[str(term).strip()] * boost_entity[str(term)] * boost_term_in_article[str(term)])
									except: pass 			
							except: pass
							cluster_score[cl] /= freq
						else: break
							
					sorted_clusters = sorted( ((v,k) for k,v in cluster_score.iteritems()), reverse=True)
					print "sorted cluster_score:"
			 		print sorted_clusters
			 		#会收集top20的聚簇，并选第一条推特的标题
			 		ntopics = 20
			 		headline_corpus = []
			 		orig_headline_corpus = []
			 		headline_to_cluster = {}
			 		headline_to_tid = {}
			 		cluster_to_tids = {}
			 		for score,cl in sorted_clusters[:ntopics]:
			 			#print "\n(cluster, freq):", cl, freqTwCl[cl]
			 			clidx = (npindL == cl).nonzero()[0].tolist()
						#cluster_centroid = X[clidx].sum(axis=0)
						#centroid_tweet = vectorizer.inverse_transform(cluster_centroid)
						#random.seed(0)
						#sample_tweets = random.sample(clidx, 3)
						#keywords = vectorizer.inverse_transform(cluster_centroid.tolist())
						first_idx = map_index_after_cleaning[clidx[0]]
						keywords = window_corpus[first_idx]
						orig_headline_corpus.append(keywords)
						headline = ''
						for k in keywords.split(","):
							if not '@' in k and not '#' in k:
								headline += k + ","
						headline_corpus.append(headline[:-1])
						headline_to_cluster[headline[:-1]] = cl
						headline_to_tid[headline[:-1]] = tids_window_corpus[first_idx]

	 					tids = []
	 					for i in clidx:
							idx = map_index_after_cleaning[i]
	 						tids.append(tids_window_corpus[idx])
	#   						try:
	#   							print window_corpus[map_index_after_cleaning[i]]
	#   						except: pass	
	 					cluster_to_tids[cl] = tids	

							
					## cluster headlines to avoid topic repetition
					headline_vectorizer = CountVectorizer(tokenizer=custom_tokenize_text, binary=True, min_df=1, ngram_range=(1,1))
					#将这个语料集转化为词频矩阵
				  	H = headline_vectorizer.fit_transform(headline_corpus)
				  	#这个tweet／term矩阵的规格
				  	print "H.shape:", H.shape
				  	#获得feature词的索引,从要素数字到要素名称的数组索引
					vocH = headline_vectorizer.get_feature_names()
				  	#print "Voc(headline_corpus):", vocH
					#转化为稠密矩阵，数字格式为float
				  	Hdense = np.matrix(H.todense()).astype('float')
				  	#计算推特间距离
					distH = pairwise_distances(Hdense, metric='cosine')
					#下面再次进行重聚类过程以消除话题碎片
	 				HL = fastcluster.linkage(distH, method='average')
					dtH = 1.0
					indHL = sch.fcluster(HL, dtH*distH.max(), 'distance')
	#				indHL = sch.fcluster(HL, dtH, 'distance')
					freqHCl = Counter(indHL)
					print "hclust cut threshold:", dtH
					print "n_clusters:", len(freqHCl)
					print(freqHCl)
				except:
					pass
					continue
				#对重聚类的结果计算score并按序输出
				npindHL = np.array(indHL)
				hcluster_score = {}
	 			for hclfreq in freqHCl.most_common(ntopics):
	 				hcl = hclfreq[0]
 					hfreq = hclfreq[1]
 					hcluster_score[hcl] = 0
					hclidx = (npindHL == hcl).nonzero()[0].tolist()
					for i in hclidx:

						hcluster_score[hcl] = max(hcluster_score[hcl], cluster_score[headline_to_cluster[headline_corpus[i]]])

				sorted_hclusters = sorted( ((v,k) for k,v in hcluster_score.iteritems()), reverse=True)
				print "sorted hcluster_score:"
		 		print sorted_hclusters
		 		#对前10个话题簇进行输出
				for hscore, hcl in sorted_hclusters[:10]:
#					print "\n(cluster, freq):", hcl, freqHCl[hcl]
	 				hclidx = (npindHL == hcl).nonzero()[0].tolist()
	 				clean_headline = ''
	 				raw_headline = ''
	 				keywords = ''
	 				tids_set = set()
	 				tids_list = []
	 				urls_list = []
	 				selected_raw_tweets_set = set()
	 				tids_cluster = []
 					for i in hclidx:
 						clean_headline += headline_corpus[i].replace(",", " ") + "//"
						keywords += orig_headline_corpus[i].lower() + ","
						tid = headline_to_tid[headline_corpus[i]]
						tids_set.add(tid)
						raw_tweet = tid_to_raw_tweet[tid].encode('utf8', 'replace').replace("\n", ' ').replace("\t", ' ')
						raw_tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', raw_tweet)
  						selected_raw_tweets_set.add(raw_tweet.decode('utf8', 'ignore').strip())
  						#fout.write("\nheadline tweet: " + raw_tweet.decode('utf8', 'ignore'))
  						tids_list.append(tid)
  						if tid_to_urls_window_corpus[tid]:
	 						urls_list.append(tid_to_urls_window_corpus[tid])
 						for id in cluster_to_tids[headline_to_cluster[headline_corpus[i]]]:
 							tids_cluster.append(id)
 							 				 	
 					raw_headline = tid_to_raw_tweet[headline_to_tid[headline_corpus[hclidx[0]]]]
 					raw_headline = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', raw_headline)
 					raw_headline = raw_headline.encode('utf8', 'replace').replace("\n", ' ').replace("\t", ' ')
 					keywords_list = str(sorted(list(set(keywords[:-1].split(",")))))[1:-1].encode('utf8', 'replace').replace('u\'','').replace('\'','')					

					#Select tweets with media urls
					#If need code to be more efficient, reduce the urls_list to size 1.	
					for tid in tids_cluster:
						if len(urls_list) < 1 and tid_to_urls_window_corpus[tid] and tid not in tids_set:
								raw_tweet = tid_to_raw_tweet[tid].encode('utf8', 'replace').replace("\n", ' ').replace("\t", ' ')
								raw_tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', raw_tweet)
 								raw_tweet = raw_tweet.decode('utf8', 'ignore')
 								#fout.write("\ncluster tweet: " + raw_tweet)
 								if raw_tweet.strip() not in selected_raw_tweets_set:
									tids_list.append(tid)
									urls_list.append(tid_to_urls_window_corpus[tid])
									selected_raw_tweets_set.add(raw_tweet.strip())
														
					try:	
						print "\n", clean_headline.decode('utf8', 'ignore')#, "\t", keywords_list

					except: pass					

					urls_set = set()
					for url_list in urls_list:
						for url in url_list:
							urls_set.add(url)
	

					fout.write("\n" + str(dtime) + "\t" + raw_headline.decode('utf8', 'ignore') + "\t" + keywords_list.decode('utf8', 'ignore') + "\t" + str(tids_list)[1:-1] + "\t" + str(list(urls_set))[1:-1][2:-1].replace('\'','').replace('uhttp','http'))
		

				window_corpus = []
				tids_window_corpus = []
				tid_to_urls_window_corpus = {}
				tid_to_raw_tweet = {}
				ntweets = 0
				if t == 4:
					dfVocTimeWindows = {}
					t = 0


	file_timeordered_tweets.close()
	fout.close()

