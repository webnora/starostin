# dōn (Old English) – clustering results (Calgary corpus)

Контексты извлечены из файлов `*.txt` в `/Users/nora/src/Starostin/calgary`. Корпус не лемматизирован; поиск выполнен по поверхностным формам парадигмы глагола dōn (без префиксальных/композитных дериватов; `ge-` включен только как регулярный показатель причастия прошедшего).

Единица контекста: сегмент между символами `#` (если сегменты содержали несколько форм dōn, каждый случай учитывался отдельно, но с тем же текстом сегмента).

Артефакты:

- `calgary2/umap_tfidf_segments.png`, `calgary2/contexts_tfidf_segments.txt`
- `calgary2/umap_ppmi_segments.png`, `calgary2/contexts_ppmi_segments.txt`
- `calgary2/don_segment_grid.txt` (подбор параметров)
- `calgary2/paradigm_don.md` (парадигма и глоссы форм)

Важно: в исходных файлах корпуса переводы отсутствуют, поэтому в файлах `contexts_*.txt` поле `Translation:` содержит только русскую глоссу формы глагола (не перевод всего контекста). Ниже приведены примеры с (1) подстрочным и (2) литературным переводом.

## TF-IDF + HDBSCAN

Параметры лучшего прогона: `min_df=1`, HDBSCAN `min_cluster_size=5`, `min_samples=1`. Получено 2 кластера; шум: 58/256 (0,227).

### Cluster 0 (10 примеров)

1) OE: `dæges and nihtes do , swa ic ne sceolde ,`
   Source: `a prayer.txt#seg64`
   Подстр.: день и ночь делаю, как я не должен
   Лит.: я день и ночь поступаю так, как не следует

2) OE: `ealra goda þe he ðe ær dyde .`
   Source: `allFragments.txt#seg13`
   Подстр.: всех благ, которые он тебе прежде сделал
   Лит.: всех благ, которые он прежде даровал тебе

3) OE: `do , drihten , cyng dædum halne ,`
   Source: `allFragments.txt#seg49`
   Подстр.: сделай, Господи, царя делами целым/здравым
   Лит.: Господи, укрепи царя в делах

4) OE: `do me wegas þine wise , drihten ,`
   Source: `allFragments.txt#seg54`
   Подстр.: сделай мне пути твои мудрыми/укажи, Господи
   Лит.: Господи, укажи мне пути Твои

5) OE: `þe ðu , drihten , dydest syððan dagas wæron ,`
   Source: `allFragments.txt#seg61`
   Подстр.: что ты, Господи, сделал с тех пор как дни были
   Лит.: что Ты, Господи, совершал издавна

6) OE: `gramra to georne , þe ic geong dyde`
   Source: `allFragments.txt#seg65`
   Подстр.: слишком охотно (из) гнева, что я юным делал
   Лит.: то, что я в юности по гневу слишком охотно совершал

7) OE: `hal do þin folc , halig drihten ,`
   Source: `allFragments.txt#seg71`
   Подстр.: целым сделай народ твой, святой Господь
   Лит.: исцели (сохрани) народ Твой, святой Господи

8) OE: `aris , drihten , nu and us ricene do`
   Source: `allFragments.txt#seg99`
   Подстр.: восстань, Господи, теперь и нам быстро сделай
   Лит.: восстань, Господи, и поспеши нам помочь

9) OE: `mildsa me , mihtig drihten , swa ðu manegum dydest ,`
   Source: `allFragments.txt#seg120`
   Подстр.: смилуйся надо мной, могучий Господь, как ты многим сделал
   Лит.: помилуй меня, Всемогущий Господь, как Ты миловал многих

10) OE: `gebæt mid his bridle , hafað butu gedon ,`
   Source: `allMeters.txt#seg180`
   Подстр.: исправил своим уздечьем, имеет искупление сделанным
   Лит.: исправил (обуздал) это и совершил искупление

### Cluster 1 (10 примеров)

1) OE: `ren forswelgan ; swa deð ricra nu`
   Source: `allMeters.txt#seg1607`
   Подстр.: дождь поглощать; так делает богатых (нрав) ныне
   Лит.: таково ныне поведение богатых: они все «поглощают»

2) OE: `gode forgylde , swa he nu gyt dyde !"`
   Source: `beowulf.txt#seg955`
   Подстр.: Бог (это) воздаст, как он теперь еще делал!
   Лит.: да воздаст Бог, как Он и ныне воздает!

3) OE: `gumena cynnes , swa he nu git deð .`
   Source: `beowulf.txt#seg1057`
   Подстр.: рода людей, как он теперь еще делает
   Лит.: для рода человеческого - как Он делает и поныне

4) OE: `gear in geardas , swa nu gyt deð ,`
   Source: `beowulf.txt#seg1133`
   Подстр.: год в ограды/дома, как ныне еще делает
   Лит.: из года в год - как это бывает и ныне

5) OE: `gumena dryhten , ðonne ic gyt dyde ,`
   Source: `beowulf.txt#seg1823`
   Подстр.: владыка людей, когда я еще делал
   Лит.: о владыка людей, когда я был еще (в силах) действовать

6) OE: `gumena gehwylcum , swa he nu gen deð .`
   Source: `beowulf.txt#seg2858`
   Подстр.: каждому из людей, как он теперь еще делает
   Лит.: каждому человеку - как Он делает и сейчас

7) OE: `geornor bigonge þonne he gen dyde ,`
   Source: `juliana.txt#seg111`
   Подстр.: усерднее служил/занимался, чем он еще делал
   Лит.: действовал еще усерднее, чем прежде

8) OE: `geomrende gast deð nu gena swa ,`
   Source: `solomon and saturn.txt#seg264`
   Подстр.: стонущий дух делает теперь снова так
   Лит.: стенающий дух и ныне поступает так же

9) OE: `lifiendum gode lofsang doð ,`
   Source: `soul and body i.txt#seg70`
   Подстр.: живущему Богу песнь хвалы делайте
   Лит.: вознесите песнь хвалы живому Богу

10) OE: `gode lifgendum lofsong doð ,`
   Source: `soul and body ii.txt#seg65`
   Подстр.: Богу живущему песнь хвалы делайте
   Лит.: воспойте хвалу живому Богу

## PPMI + HDBSCAN

Параметры лучшего прогона: `min_df=2`, HDBSCAN `min_cluster_size=5`, `min_samples=1`. Получено 2 кластера; шум: 14/256 (0,055).

### Cluster 0 (10 примеров)

1) OE: `dæges and nihtes do , swa ic ne sceolde ,`
   Source: `a prayer.txt#seg64`
   Подстр.: день и ночь делаю, как я не должен
   Лит.: я день и ночь поступаю так, как не следует

2) OE: `do , drihten , cyng dædum halne ,`
   Source: `allFragments.txt#seg49`
   Подстр.: сделай, Господи, царя делами целым/здравым
   Лит.: Господи, укрепи царя в делах

3) OE: `do me wegas þine wise , drihten ,`
   Source: `allFragments.txt#seg54`
   Подстр.: сделай мне пути твои мудрыми/укажи, Господи
   Лит.: Господи, укажи мне пути Твои

4) OE: `hal do þin folc , halig drihten ,`
   Source: `allFragments.txt#seg71`
   Подстр.: целым сделай народ твой, святой Господь
   Лит.: исцели (сохрани) народ Твой, святой Господи

5) OE: `aris , drihten , nu and us ricene do`
   Source: `allFragments.txt#seg99`
   Подстр.: восстань, Господи, теперь и нам быстро сделай
   Лит.: восстань, Господи, и поспеши нам помочь

6) OE: `mildsa me , mihtig drihten , swa ðu manegum dydest ,`
   Source: `allFragments.txt#seg120`
   Подстр.: смилуйся надо мной, могучий Господь, как ты многим сделал
   Лит.: помилуй меня, Всемогущий Господь, как Ты миловал многих

7) OE: `swa doð wudufuglas ; þeah hi wel sien ,`
   Source: `allMeters.txt#seg328`
   Подстр.: так делают лесные птицы; хотя они хорошо (живут/бывают)
   Лит.: так поступают и лесные птицы, хотя им и хорошо

8) OE: `hio sceal eft don þæt hio ær dyde ,`
   Source: `allMeters.txt#seg372`
   Подстр.: она должна снова делать то, что она прежде делала
   Лит.: ей снова придется повторить то, что она делала раньше

9) OE: `weoruldwelena ; hi þæt wel doð ,`
   Source: `allMeters.txt#seg496`
   Подстр.: (о) мирских богатствах; они это хорошо делают
   Лит.: (стремясь) к мирским благам, они умеют это делать

10) OE: `ðeah hine se dysega do to cyninge ,`
   Source: `allMeters.txt#seg399`
   Подстр.: хотя его глупец делает в царя (возводит в царя)
   Лит.: пусть даже глупец возведет его в цари

### Cluster 1 (в кластере всего 6 контекстов - привожу все)

1) OE: `drihten si gebletsad , þe þæt ne dyde æfre ,`
   Source: `allParis.txt#seg1778`
   Подстр.: Господь да будет благословен, который то не сделал никогда
   Лит.: да будет благословен Господь, который никогда так не поступал

2) OE: `þonne drihten wyle gedon æfter ,`
   Source: `allParis.txt#seg1810`
   Подстр.: когда Господь захочет сделать после/далее
   Лит.: когда Господь пожелает совершить это впоследствии

3) OE: `drihten si gebletsad , þe he ne dyde æfre`
   Source: `allParis.txt#seg3367`
   Подстр.: Господь да будет благословен, который он не сделал никогда
   Лит.: да будет благословен Господь: Он никогда этого не делал

4) OE: `drihten æfre , dyde swa he wolde ,`
   Source: `allParis.txt#seg4947`
   Подстр.: Господь всегда делал так, как он хотел
   Лит.: Господь всегда поступал по Своей воле

5) OE: `dior dædfruma , gedon wolde`
   Source: `beowulf.txt#seg2089`
   Подстр.: смелый совершитель деяний сделать хотел
   Лит.: доблестный герой хотел это совершить

6) OE: `drihten wedera gedon wolde ;`
   Source: `beowulf.txt#seg2185`
   Подстр.: владыка ветров сделать хотел
   Лит.: Владыка (бурь/небес) так задумал
