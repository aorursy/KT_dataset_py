# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# !pip install tweepy
# !pip install pigeon-jupyter
# !pip install vadersentiment
# !pip install textmining
# !pip install nltk

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tweepy 
import nltk.stem as stemmer
from tweepy import OAuthHandler
import json
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from bs4 import BeautifulSoup
import re, string, unicodedata


def limpiar_text(texto):
    text_split= texto.split()
    for palabra in text_split:
        if(len(palabra) > 1 and palabra[0:1].upper()=='@'):
            texto=texto.replace(palabra, " ");
        if(len(palabra) > 1 and palabra[0:1].upper()=='#'):
            texto=texto.replace(palabra, " ");
        if(len(palabra) > 4 and palabra[0:4].upper()=='HTTP'):
            texto=texto.replace(palabra, " ");
        
    texto = re.sub('[^a-zA-Z0-9 \n\.]', '', texto)
    return texto.strip();


print(limpiar_text("#Deportes. Los #$##%#$invito a escucharnos desde las 7pm en @RadNalCo. https://t.co/UUDgpSVRfo"));


conjunto_entrenamiento=[{"text":"La música habla. Todas las vidas importan. Hoy en SenalClasica, música que se levanta contra el racismo, contra la violencia, contra la discriminación. Los invito a escucharnos desde las 7pm en RadNalCo.",
"label":"positivo"},
{"text":"Se puede llamar a todo el mundo sin clasificar,entre feministas hombres y mujeres, eso también clasicismo y promueve el racismo y la violencia. Si hablamos de unión, entenderíamos que la violencia en general y sin importar la víctima,es el verdadero racismo es la violencia.",
"label":"neutro"},
{"text":"Los Latinos de EEUU se quejan que los afros los discriminan por su Raza. Un peleador de SFC dice que debia pelear contra ellos por el racismo hacia el",
"label":"negativo"},
{"text":"MiguelPoloP INEPTO no fue contra Trump fue contra el RACISMO, dónde putas tiene la capacidad de contextualizar una temática.",
"label":"negativo"},
{"text":"O Racismo e a pandemica do mundo...El Racismo es una pandemica mundial.",
"label":"neutro"},
{"text":"El racismo y la brutalidad policial asesinaron a Anderson Arboleda en Colombia, contribuye para que su caso no quede en la impunidad.",
"label":"neutro"},
{"text":"mi papá es como uds, full motivado está, q el racismo está mal, q el es negro también, mejor dicho tiene puesta la 10, pero cuando pasa alguna situación aquí y un grupo de personas decide pronunciarse, ahí si: vándalos hp, mamado de los jóvenes, gamines",
"label":"negativo"},
{"text":"El racismo también es una pandemia ","label":"neutro"},
{"text":"Bueno con estas protestas a nivel mundial contra el racismo sin duda vendrá una segunda ola de contagios.",
"label":"negativo"},
{"text":"Ahora que sufrieron daño los Mercedes-Benz no habrá racismo?",
"label":"neutro"},
{"text":"El racismo es una pandemia también.","label":"neutro"},
{"text":"Tuvimos una edición especial dedicada a la discriminación en el país. Estos fueron los análisis en estas ciudades capitales: La lucha contra el racismo escondido en Medellín y Barranquilla",
"label":"neutro"},
{"text":"Buenos días.mucho embrujo,hechizo o encantamiento que te han hecho que no te disgusta el olor a mierda de tu enemigo en cambio no  soportas tu propio olor ni el de tus hijos ,tu familia,tu raza .porque el no gustarte otras mierdas eso no es racismo ni discrimacion ,eso es normal!",
"label":"negativo"},
{"text":"Que los gringos salgan a marchar en contra del racismo, no quita que sigan siendo xenófobos con ustedes por latinos, a ver si algún día dejamos de idolatrarlos",
"label":"negativo"},
{"text":"Veremos como se comporta el virus en EEUU, Si las protestas disparan los contagios o no. El racismo en el 2020 no puede aflorar después de estar erradicado en el mundo. Art 13 Constitución política",
"label":"neutro"},
{"text":"Pero si no tiene que ver. Imagínate que pusieran a Mulán latina, es cuestión que se parezca a la referencia no racismo.",
"label":"neutro"},
{"text":"Lo que veo es una niña que han educado para que entienda su realidad a tiempo defiende los derechos ganados y siga luchando por los derechos que hacen falta. El racismo es una realidad Así la quieran negar y está bien que los chiquitos desde temprana da entiendan lo que sucede.",
"label":"neutro"},
{"text":"Eso ya pasó de Moda, lo de ahora es el Racismo",
"label":"neutro"},
{"text":"Racismo, el detonante de una crisis institucional en EE. UU. ",
"label":"neutro"},
{"text":"video en Nueva York, no al racismo, no a la discriminación, igualdad de oportunidades. SOMOS CREACIÓN PERFECTA DE DIOS (TODOS LOS SERES HUMANOS). VALOREMONOS.",
"label":"positivo"},
{"text":"Omitir la calidad del aire como covariable en un estudio de COVID-19 y raza no solo es un error sino que desconoce que más que la raza, es el racismo el que está detrás de las diferencias.",
"label":"neutro"},
{"text":"Interesante la Colombia Abyecta Domestica Felonica  Traquetica ....No reacciona....Por la inclusión y el racismo todos  a la Calle a Protestar y Exigir ",
"label":"negativo"},
{"text":"No más discriminación, no mas racismo, no mas abusos!",
"label":"neutro"},
{"text":"Odiar a animales no es racismo, es rencor hacia esos hp nefastos...",
"label":"negativo"},
{"text":"@Interesante porque Canadá se debate entre lo multicultural y el racismo.",
"label":"neutro"},
{"text":" 1. Se escribe hombre 2. Si, tengo un prototipo, No son hombres blancos. 3. El hecho de que no le guste no le da derecho a decir que está desorganizado, porque sus comentarios y estereotipos generan racismo y EL RACISMO MATA!! FIN DE LA CONVERSACIÓNHASTA LUEGO",
"label":"negativo"},
{"text":"EEUU Despierta contra el Racismo y el NeoFacismo.",
"label":"neutro"},
{"text":"Yo digo NO al racismo.","label":"positivo"},
{"text":"Género = Capitalismo ,Especie = racismo, machismo, homofobia, Xenofobia Etc.",
"label":"neutro"},
{"text":"Muchisimas personas en el mundo protestan por el racismo que llevó a un crimen de odio en EEUU, en Colombia dia a día ocurren hechos de racismo de los que la gente guarda un silencio complice.AndersonArboleda murió asesinado a bolillazos como lo describe UnCaricaturista",
"label":"negativo"},
{"text":"Merecido cuando la mayoría que protesta en contra del racismo y otras injusticias no vota, o torpemente vota por el que diga Uribe",
"label":"negativo"},
{"text":"El verdadero racismo es la violencia ,ella no discrimina",
"label":"neutro"},
{"text":"¿De verdad cree que la ola de violencia y vandalismo son manifestaciones contra el racismo?Lo creía inteligente.",
"label":"negativo"},
{"text":"El racismo, al igual que la eterna lucha de clases, es la perpetuación del victimismo.En pleno siglo XXI le retó a nombrar un solo artículo de alguna constitución democrática, donde se segregue o discrimine.",
"label":"neutro"},
{"text":"1.Sobre el despertar contra del racismo y las protestas, a mi juicio lograron el objetivo mas importante, procesar a responsables allá no acá.Nuestra carrera evolutiva es larga, medios excluyentes a toda minoría, religiones y deidades que sutilmente a través de la historia",
"label":"neutro"},
{"text":"La receta para curar la lgbtfobia, el machismo, el racismo y la ignorancia en general es la educación.",
"label":"positivo"},
{"text":"Están los que han encontrado en ayudar el verdadero sentido de la vida, a sensibilizado más a las personas hacia temas que siempre han existido y simplemente no eran transcendentales ( por ejemplo, racismo), a unido a familias que hace mucho tiempo no compartían momentos juntos",
"label":"positivo"},
{"text":"Y en Colombia hay racismo y matan a líderes etc y la gente ni se indigna aún q nos pasa como sociedad ah",
"label":"negativo"},
{"text":"Michael Jordan donará 100 millones de dólares para la lucha contra el racismo",
"label":"positivo"},
{"text":"Momento. Uno no se burla de esa gente, uno se indigna de esa gente. Y no tiene que ver una mondÁ, el racismo con la estafadora engaña pingos que mencionas",
"label":"negativo"},
{"text":"No hemos superado el racismo, la xenofobia, el machismo, las guerras y quieren que vengan aliens a la tierra? ",
"label":"neutro"},
{"text":"Así o más claro?Es violencia, no racismo. Violencia es violencia. Hay más asesinatos de negros contra blancos y de negros contra negros, de chinos contra chinos y pare de contar.",
"label":"neutro"},
{"text":"Trudeau hincó una rodilla durante una manifestación en contra del racismo",
"label":"neutro"},
{"text":"El racismo es un virus mayor ",
"label":"neutro"},
{"text":"Ratificas tu Discriminación y Racismo excluyendo a la población Negra y Afrocolombiana del plan de desarrollo del municipio de Bucaramanga.. Violador de derechos humanos étnicos",
"label":"negativo"},
{"text":"MiguelPoloP Indolente esclavo por ningún lado hay RACISMO.","label":"negativo"},
{"text":"De verdad que yo no logro comprender la homofobia y el racismo. Literalmente no se que es lo que puede pensar alguien en su idiotez de sentirse superior.",
"label":"negativo"},
{"text":"Soy ciego, violento, carezco de lectura y además padezco racismo introyectado. Estos twitteros resultan ser de todo, incluso psicólogos.Para tu consuelo, si requiero una terapia, te busco luego",
"label":"negativo"},
{"text":"¿Y para cuando el estallido social en rebeldía se levantara en Colombia contra la política fascista?",
"label":"negativo"},
{"text":"Estoy feliz porque una de mis fotos está siendo utilizada para trasmitir un mensaje en contra del racismo",
"label":"positivo"},
{"text":"Los ñeros Petroguisos hablando de racismo. Es que de todo saben. Jajajaja jajajaja payasos incoherentes",
"label":"negativo"},
{"text":"ya se le olvidó su racismo confesó? Lea usted sus obras y milagros CINICO . ahoea el adalid de la causa de Discriminación Racial. Que horror!",
"label":"negativo"},
{"text":"el racismo en Colombia es histórico y estructural",
"label":"negativo"},
{"text":"Cual racismo!!! Lo que son es unos ladrones de miedo. Vaya a ese pueblo para que vea. Es rico, pero toda se la roban los políticos de allá con la anuencia de los senadores y todas las ias. Allá no se salva nadie. Todos ladrones.",
"label":"negativo"},
{"text":"Si, en efecto, todos los negros hemos vivido en carne propia el racismo desde que somos niños. A los niños los educan así, ellos no nacen creyendo que los negros somos malos, menos inteligentes o más feos que cualquier otro blanco, el fondo de esto es mucho más amplio.",
"label":"negativo"},
{"text":"Ahora sí los políticos se rasgan las vestiduras por las condiciones de vida del Chocó. Tuvo que llegar una pandemia para que ellos se dieran cuenta que el departamento existe, que allí existe gente... eso también es racismo y discriminación.",
"label":"negativo"},
{"text":"Y por supuesto! esto no lo entendera al que no le pase, incluso este tiktok si les parecera estupido.! eso es racismo. marcado en lo mas minimo de sus conocimientos",
"label":"neutro"},
{"text":"Y esta misma estigmatización lo que lleva es al racismo, la xenofobia, la homofobia, etc que por tanto tiempo hemos vivido, nunca se ha ido, siempre ha estado ahi presente!",
"label":"negativo"},
{"text":"Ahí no aplica la muerte por racismo, si es lo que quieres decir. Son muertes de una guerra absurda, en la que específicamente con lo de  Bojayá, estás narrando la mitad de esa historia. Te invito a q te documentes de la historia completa.",
"label":"neutro"},
{"text":"El racismo no importa todos tenemos el mismo derecho hacia el estado gobierno nacional.",
"label":"neutro"},
{"text":"Rodrigo es cierto d las injusticias q han debido pasar a través del tiempo, el racismo no debería existir, eso se puede minimizar enseñando en casa a nuestros hijos sobre el tema, y familiares. El resentimiento, estereotipos y demás interfieren entre ellos mismos xa minimizarlo",
"label":"neutro"},
{"text":"Me pregunto si comportarnos frente a alguien de manera despectiva, que no aparenta ser de cierta clase socio económica,  también es racismo?",
"label":"neutro"},
{"text":"Yo esperando que invitaran a alguien que hablara sobre la política penal en EE.UU y el racismo, pero es que me pasó de ilusa.",
"label":"neutro"},
{"text":"Y en la India, los templos se aprovechan de las mujeres que ofrenda su pelo, al lucrarse y no pagarles a sus propietarias. Es la manera como el capitalismo explota el racismo hacía las mujeres afro y la pobreza de otras comunidades.",
"label":"negativo"},
{"text":"Cata la verdad es que no creo que sea racismo. Sus gobernantes, todos nacidos en las familias políticas de ese Departamento, han tenido índices de corrupción dignos del libro de Guinness Récord; Roy Barreras se sentiría orgulloso de ellos.",
"label":"neutro"},
{"text":"No está bien el empujon, pero ya son muchos días jodiendo en la calle y ese cuento del racismo y que les dolió la muerte del negro eso no se los cree nadie , ya se sabe q les están pagando para dañar , robar y joder el Gno Trump",
"label":"negativo"},
{"text":"La pelea contra el racismo y el abuso policial comienza en NUESTRAS comunidades y en NUESTROS países. Un cubito negro por lo que pase en el primer mundo no nos quita la responsabilidad de afrontar esos problemas AQUI en Latinoamérica",
"label":"neutro"},
{"text":"Aquí en Colombia el racismo es mínimo, si el vecino te hace el feo el otro vecino no, las naciones se tienen que disculpar con África y las negritudes del mundo porque a lo largo de la historia son los que mas han sufrido y los que han tenido que llorar en silencio.",
"label":"neutro"},
{"text":"Las revistas mexicanas de moda y farándula se me hacían terriblemente vacías. Pero su racismo y clasismo no tienen parangón.",
"label":"negativo"},
{"text":"Como nos aguantamos lo de Venezuela, pues aguantémonos el racismo, dejen de llamar la atención con las protestan en contra de cientos de años de crímenes. Cordial invitación de @vanesavallejo3 Payasa, utilizar a Venezuela para minimizar el racismo, debería darle vergüenza!",
"label":"negativo"},
{"text":"ClaudiaLopez Luego de dos cuatro horas de protestas se reactivo el comercio en Corabastos y a esta ahora avanza la reunion entre comerciantes y funcionarios de la Alcaldia de Bogota y del Gobierno Nacional buscan llegar a un acuerdo ante la decision de cerrar cuatro bodegas de esa central de abastos","label":"positiva"},
{"text":"NoticiasCaracol Deberiian contratarlo en la alcaldia de Bogota para que apliquen medidas estrictos poco y placa pico y cedula toque de queda ley seca multas al doble","label":"negativa"},
{"text":"Cero y van Dos hay que hacerlo pasar por la alcaldia de Bogota ahi se queman","label":"negativa"},
{"text":"Gracias al trabajo conjunto con la Alcaldia de Bogota dos nueve siete familias que vivian en alto riesgo ahora tendran una vivienda propia y digna uno nueve nueve de estos hogares contaron con subsidios del Gobierno Nacional","label":"positiva"},
{"text":"El fracaso de ClaudiaLopez Bogota esta desorientada confundida y con una gran probabilidad de contagio mientras la alcaldesa sigue en la oficina planeando la campana presidencial La alcaldesa irresponsable","label":"negativa"},
{"text":"En dos cero uno seis envie un oficio a Alcaldia Bogota sobre una ocupacion indebida del espacio publico me acaba de llegar la respuesta Sin disculparse por la tardanza me responden que cuando pase el estado de excepcion declarado por esta emergencia haran el operativo cuatro anos despues","label":"negativa"},
{"text":"Con las medidas implementadas por la Alcaldia de Bogota se ha evitado aglomeracion en la zona de San Victorino Tambien se regulo la venta informal en el espacio publico por la emergencia del COVIDuno nueve","label":"positiva"},
{"text":"Buenos dias sigo escribiendo para lograr la ayuda de la alcaldia Bogota ClaudiaLopez o de un vete para mis chiquis De nuevo solicito su ayuda para retuit Quiero apelar al carino que tiene ClaudiaLopez por los animales Anoche volvi a quedarme tarde con Sachi","label":"negativa"},
{"text":"Ustedes diciendo en las elecciones a la alcaldia que tocaba votar por el menos peor jsjsjajaj hay que tener dos dedos de frente para saber que Claudia Lopez no era una opcion para Bogota no vieron entrevistas debates toda la mierda que hablo Que tristeza tan hpta","label":"negativa"},
{"text":"En estas manos esta la Alcaldia de Bogota","label":"negativa"},
{"text":"Cual es la evidencia cientifica que avala seguir con esta estupidez","label":"negativa"},
{"text":"Hoy la alcaldesa ClaudiaLopez revisa balance de las acciones realizadas esta semana desde el Puesto de Mando Unificado en la Alcaldia Local de Kennedy Hemos realizado pruebas entregado mercados transferencias monetarias Todos estamos cuidando a Kennedy","label":"positiva"},
{"text":"Es que la alcaldia de ClaudiaLopez le entregara mercados de alimentos perecederos en estado de descomposicion a familias de vendedores ambulantes de Bogota","label":"negativa"},
{"text":"Para cuando la fiscalia le pasara la factura de cobro a Petro por sus malos manejos en la alcaldia de Bogota","label":"negativa"},
{"text":"ClaudiaLopez Bogota Tal cual los de abastos no se dejaron ver las hue Y a la alcaldia le toco agachar la cabeza","label":"negativa"},
{"text":"deberia tambien contar las verdades de la desfalcada tan brava que usted le metio a Bogota en su alcaldia","label":"negativa"},
{"text":"quizas viste la noticia sobre una supuesta contratista de la Alcaldia de Bogota con capota negra y tapabocas ordenando los desalojos en Ciudad Bolivar en medio de la pandemia Resulta que al parecer ella misma cobraba a las personas que vivian en esas invasiones","label":"negativa"},
{"text":"La rinde todos los dias en la pagina de la alcaldia si tanto le interesa lea un poco sobre el tema","label":"negativa"},
{"text":"InfoBogota Como pagar por cuotas el impuesto a vehiculos y el predial en Bogota Se amplio el plazo de pago para aliviar la economia de los contribuyentes en medio de la pandemia","label":"positiva"},
{"text":"La GobBoyaca junto con la Alcaldia de Bogota Gobernacion de Cundinamarca y representantes de Corabastos durante la noche de este viernes cinco de junio lograron un acuerdo que evitara el cierre de esta central mayorista","label":"positiva"},
{"text":"ClaudiaLopez desde el segundo cargo mas importante de la nacion PRETENDER hacer oposicion campana presidencial y administrar bien la alcaldia de BOGOTA es un ROBO contra sus electores","label":"negativa"},
{"text":"Esa vieja cree que la Alcaldia de Bogota es la cueva de Ali baba Es que deberian destituirla y presentar los contratos a los entes de control a ver cuales pasan y cuales no","label":"negativa"},
{"text":"Si la Alcaldia y la Nacion han anunciado alivios en los cobros por que razon se estan quejando los usuarios del servicio de aseo","label":"negativa"},
{"text":"Dos miradas al Plan de Desarrollo de Claudia Lopez","label":"positiva"},
{"text":"Pero se esperaba no era para preparar mas camas de UCI que debia trabajar y gestionar la alcaldia de Bogota","label":"negativa"},
{"text":"Si la Alcaldia y la Nacion han anunciado alivios en los cobros por que razon se estan quejando los usuarios del servicio de aseo","label":"positiva"},
{"text":"ClaudiaLopez Eso es como echarle la culpa al gobierno por la ineptitud y los fracasos de la alcaldia para controlar la pandemia en Bogota","label":"negativa"},
{"text":"Si la Alcaldia y la Nacion han anunciado alivios en los cobros por que razon se estan quejando los usuarios del servicio de aseo","label":"negativa"},
{"text":"Aunque la alcaldia de Bogota y el Gobierno Nacional han anunciado beneficios para los servicios publicos estos no estarian llegando como se esperaba Autoridades responden a esta situacion","label":"negativa"},
{"text":"ClaudiaLopez Que vaya a Corabastos y a Kennedy y haga socializacion del buen manejo de la Alcaldia de Bogota y de seguro la sacan vitoriada en hombros a esta ilustre representante del mamertismo anticorrupcion","label":"negativa"},
{"text":"Dos miradas al Plan de Desarrollo de Claudia Lopez","label":"positiva"},
{"text":"Dos miradas al Plan de Desarrollo de Claudia Lopez","label":"positiva"},
{"text":"FiscaliaCol que se ha avanzado de la investigacion de las familias de concejales congresistas con contratos a dedo en la alcaldia de Bogota Respuesta Incriminar a Duque Capturar a nadie Perdida millonaria del erario","label":"negativa"},
{"text":"Deje de desviar la atencion de su fracaso en la alcaldia y mejor trate de resarcir en algo su ineficiencia en el manejo del virus o sino Bogota terminara siendo el epicentro de la pandemia en el hemisferio","label":"negativa"},
{"text":"ClaudiaLopez Bogota Tuvieron que pasar dos meses para que la Alcaldia se diera cuenta que uno de los mayores puntos de aglomeracion es ese Prueba de la infinita irresponsabilidad e ineficiencia de la pequena tirana popularperoineficiente","label":"negativa"},
{"text":"MACHO DE PUEBLO SI QUIERE SER POLITIQUERA LE RECOMIENDO QUE LO HAGA RENUNCIE A LA ALCALDIA POR SU MEDIOCRIDAD NO PUEDE CON BOGOTA DEDIQUESE A SU MUNECA Y A DISFRUTAR DE LAS MIELES DEL DINERO MAL HABIDO","label":"negativa"},
{"text":"ClaudiaLopez Preocupase por Bogota que por culpa suya esta vuelto mierda creo que es mejor que renuncie y se valla a dirigir una alcaldia de una vereda porque de administracion preparacion estudio y liderazgo esta en ceros con que ideas piensa manejar un pueblo con ideas revolucionarias","label":"negativa"},
{"text":"De cinco ocho muertes en el pais por Covid uno nueve la mayor concentracion de ellas se produjeron en Barranquilla con uno siete fallecimientos Ahora van a decir que Bogota ha hecho mal el control contra el Covid uno nueve en la alcaldia de ClaudiaLopez Admirable es el trabajo de la alcaldesa","label":"positiva"},
{"text":"En las medidas hay aciertos y desaciertos tanto del gobierno como de la Alcaldia Pero en Bogota no pasa nada con la indisciplina Que control se hace Ninguno Y mas cuando la alcaldesa da via libre para que informarles salgan Las medidas deben adaptarse a la realidad","label":"negativa"},
{"text":"ClaudiaLopez La Alcaldesa Claudia Lopez ha sido un fracaso en la Alcaldia de Bogota no fue capaz con Corabastos tuvo que recurrir a funcionario del Gobierno","label":"negativa"},
{"text":"ClaudiaLopez A Esta inepta le quedo grande la alcaldia de Bogota y le quiere echar la culpa a los demas a duque que es el que mejor lo a echo junto al alcalde de medellin","label":"negativa"},
{"text":"Esta asustada Ojala la Fiscalia vigile con lupa los millonarios contratos de la alcaldia de Bogota firmados por usted alcaldesa izquierdista comunista apenas cumpla el primer ano de gobierno vamos a revocar su mandato populista usted salio peor que Petro","label":"negativa"},
{"text":"No creo que a ningun otro alcalde de Bogota le haya tocado trabajar tanto tan rapido y en contra hasta de la misma naturaleza como le ha tocado a ClaudiaLopez le toco organizar todos los problemas de la ciudadania en sus primeros seis meses de alcaldia","label":"positiva"},
{"text":"ClaudiaLopez Oiga dona chimoltrufia ud esta liderando la alcaldia de bogota o esta haciendo campana para el dos cero dos dos Atrevida utilizar Bogota para hacer campana","label":"negativa"},
{"text":"Y los municipios tampoco participan antes de hacer la ley organica Todo queda en la Gobernacion y Alcaldia de Bogota","label":"negativa"},
{"text":"La izquierda de ratas es el cancer de colombiano miren a samuel moreno miren y a petro con la alcaldia de bogota pura corrupcion lo que pasa es que ahi personas ciegas arrodilladas a los delincuentes asesinos violadores claro que demas que son de los mismos","label":"negativa"},
{"text":"ClaudiaLopez Bogota Expropiese cierrese clausurese Asi resuelve todo esta alcaldia chavista Luego cuando la gente se rebota y le toca recular y retractarse de lo dicho le llaman acuerdo historico a favor de los campesinos","label":"negativa"},
{"text":"La alcaldia de la loca permite las marchas de Fecode pero no la oracion en las iglesias","label":"negativa"},
{"text":"Buen hilo y cuestionamientos para evaluar Sin embargo la Alcaldia de Bogota prometio cuatro cero cero cero UCI para el dos siete cero cinco hoy cero cinco cero seis estamos apenas rondando las uno cero cero cero Si el deficit del gobierno nacional son cinco cero cero donde estan las dos cinco cero cero que faltan para la meta","label":"negativa"},
{"text":"Para desviar la atencion del caos en que tiene a Bogota ClaudiaLopez se la pasa trinando contra Duque Decidio volver al gobierno como chivo expiatorio de los fracasos de su alcaldia","label":"negativa"},
{"text":"Se delican por no compartir datos personales a la Alcaldia pero le regalan toda su informacion a hackers con sus Que tipo de Pokemon soy segun mi foto","label":"negativa"},
{"text":"Que opinara el FARO MORAL que ocupa la Alcaldia de Bogota","label":"negativa"},
{"text":"La alcaldia de Bogota nos dice que si no aceptamos las condiciones de ellos nos traen el ESMAD","label":"negativa"},
{"text":"ClaudiaLopez Ud deberia estar trabajando no metiendose en chismes de departamentos ajenos por eso es que Bogota es un circo en estos momentos con su payasada de alcaldia","label":"negativa"},
{"text":"La alcaldia de Bogota esta aca para hacer un tratado un tratado de llevarse uno tres cero personas","label":"negativa"},
{"text":"El trabajo en casa se torna difícil Cara de reloj a las seis en puntoLevántate y arréglate como si fueras a salir, eso despierta tu mente. Acondiciona un espacio para realizar tu trabajo Evita distracciones y alteraciones así tendrás un día productivo",
"label":"positivo"},
{"text":"En Colombia el empleador que termina el contrato de trabajo sin justa causa debe indemnizar al trabajador, pero si es el trabajador el que termina el contrato sin una justa causa no debe indemnizar. Te dejan el trabajo tirado y no pasa nada. Que vulgaridad de país.",
"label":"negativo"},
{"text":"Una de las características del socialismo es que una élite de unos pocos que se financia con los impuestos y el producto del trabajo de todos, se enquista en el poder para gobernar en favor de si mismos y de sus amigos. En Colombia es peor por que el líder es un narcomatarife",
"label":"negativo"},
{"text":"Nuestro trabajo humanitario no se ha detenido ni un minuto. En medio de la COVID19, nuestros colegassiguen llevando ayudas a comunidades afectadas por el conflicto o que no tienen acceso a elementos de aseo para prevenir esta pandemia. Les deseamos un feliz día desde Caquetá.","label":"neutro"},
{"text":"Por mensajes conciliadores con los comunistas Juan Manuel Santos destruyó todo el trabajo de erradicación del terrorismo en Colombia y esos mensajes dieron pie a que jefes de la guerrilla genocida estén libres y ocupando curules en el Congreso Colombiano. Eso es Justo?",
"label":"negativo"},
{"text":"Yo si viví en Colombia, en Bogotá para ser más exacto, y de echo no me aceptaron el permiso de trabajo allí y eso que iba con contrato de trabajo, y no he sufrido más racismo en mi vida como el que sufrí allí, racismo d e gente con apellidos como, Fernández, Gómez, García...",
"label":"negativo"},
{"text":"Nueva vacante en Manizales! Si tienes experiencia en el sector industrial y ferretero como asesor de ventas, es tu oportunidad.",
"label":"positivo"},
{"text":"los jugadores realizaron trabajo físico, ejercicios con balón, remates a portería y disputaron un partido. Nacho hizo con el grupo la primera parte de la sesión y después siguió en solitario sobre el césped. Jovic continúa con su proceso de recuperación.",
"label":"neutro"},
{"text":"realiza la medición de desempleo bajo varios indicadores, BUSCAMOS PROMOVER GARANTÍAS AL TRABAJO, tienes derechos, ASESÓRATE CON NOSOTROS, sé parte de población ocupada y disminuye el riesgo de vulnerabilidad económica",
"label":"neutro"},
{"text":"Hay mucho resentido en Colombia con la cabeza mal, que se alegra si se quiebran las empresas, por eso los populistas como la López y otros, consiguen apoyo aunque lleven al país la debacle. La López no quiere normalizar el trabajo. Le está sacando provecho a la crisis.",
"label":"negativo"},
{"text":"Un especial reconocimiento a la Seccional por su liderazgo y trabajo permanente para la competitividad, desarrollo social y económico de la región en estos 75 años.",
"label":"positivo"},
{"text":"Me siento muy honrado de que me incluyan en la Red. Tenemos que seguir adelante en ese trabajo de adaptación que tiene que tener el SINA con los jóvenes",
"label":"positivo"},
{"text":"Aclarando que esto conlleva a entender que este proceso no tiene ninguna implicación con respecto a las relaciones laborales, por lo que no existirán afectaciones, desmejoras o terminaciones de los contratos de trabajo de sus colaboradores",
"label":"neutro"},
{"text":"Y si las fuerzas militares de Colombia están haciendo tan buen trabajo últimamente, cual es la necesidad de traer tropas extranjeras, en fin la corrupción, perdón la hipocresía.",
"label":"negativo"},
{"text":"Felicitaciones May Muchos años de trabajo y se merecen celebrar este día. Abrazos desde ColombiaTenemos una gran riqueza en biodiversidad en el país. Cada día se integra muchísimo más el trabajo articulado con las Autoridades",
"label":"positivo"},
{"text":"Excelente trabajo . La política en Colombia es una empresa corrupta, donde se invierten unos millones en apoyar un candidato y luego se devuelven en contratos multimillonarios a los inversores, sus amigos y negocios con abogados y periodistas que cubren sus andanzas.",
"label":"negativo"},
{"text":"Es realmente importante estar frecuentemente informados sobre todo lo que está pasando con nuestros equipos de trabajo y desarrollar relaciones de confianza internamente en la organización",
"label":"neutro"},
{"text":"El sondeo Una aproximación al Covid 19 y su incidencia en el mundo del trabajo doméstico en Colombia realizado por 16 organizaciones que trabajan por los derechos de las trabajadoras domésticas titulado , recogió testimonios de las 678 mujeres que lo respondieron.",
"label":"neutro"},
{"text":"reconocemos la labor de los miles de campesinos de Colombia; por ello seguiremos promoviendo la dignificación de su trabajo.",
"label":"neutro"},
{"text":"Porque apoyamos al campo colombiano y a los millones de hombres y mujeres que con el trabajo de sus manos y el amor por la tierra, logran la producción y desarrollo de la base de la economía del país, a ellos les decimos: GRACIAS!",
"label":"positivo"},
{"text":"trabajamos para dignificar el trabajo de los campesinos y campesinas de Colombia. Con voluntad gubernamental podremos desarrollo social a la Colombia rural.",
"label":"positivo"},
{"text":"esperamos un trabajo serio y una ayuda eficaz para tratar el flagelo del narcotráfico en Colombia",
"label":"neutro"},
{"text":"Una persona en colombia ya no puede percibir doble remuneración del estado eso es viejo .y si los maestros recibían dos era por qué les tocaba trabajar en dos y hasta tres colegios era lo justo por su trabajo.",
"label":"negativo"},
{"text":"Por  primera vez uno de los objetivos del Plan Estratégico Sectorial es la transformación digital y nosotros desde la AUNAP estamos aportando a través del trabajo para innovar y  acercando a nuestros usuarios al mundo virtual.",
"label":"positivo"},
{"text":"Me encanta vuestro trabajo os doy mi apoyo desde España","label":"positivo"},
{"text":"Ha habido varios proyectos de ley sobre el trabajo por horas y nosotros apoyamos la medida desde la empresa. Es necesario que en Colombia regulen a los trabajadores de la economía colaborativa,Camilo Sarasti, country manager iFood en Colombia",
"label":"neutro"},
{"text":"En el proyecto Rumichaca-Pasto, tomamos las medidas necesarias para velar por la salud de nuestros colaboradores. Mediante procesos de desinfección constante de los vehículos, maquinaria, herramientas y zonas de trabajo. ",
"label":"neutro"},
{"text":"Una de las cosas que he amado de mi trabajo actual es poder conocer toda la diversidad cultural en Colombia. La armonización que las autoridades indígenas acaban de hacer a una reunión me trajo la paz que hoy necesitaba mi mente",
"label":"positivo"},
{"text":"Otro caso es el de Venezuela y Colombia, donde la gente habiendo perdido el trabajo y teniendo pocas redes acá, con hijos en sus países de origen, están queriendo regresar. Colombia abrió un vuelo el 8 de junio, en el caso d Venezuela no hemos tenido noticias",
"label":"neutro"},
{"text":"Además de cultivar alimentos, sus manos siembran paz y esperanza. Hoy queremos reconocer el trabajo de todos los campesinos de Colombia que ayudan a fortalecer la seguridad alimentaria de nuestro país.",
"label":"positivo"},
{"text":"Hoy terminan tres años de trabajo en una empresa en donde fui inmensamente feliz, dónde crecí profesional y personalmente. Hoy me convierto en un número más en el porcentaje de desempleo de Colombia por cuenta de la pandemia. Muy triste todo",
"label":"negativo"},
{"text":"Tras las dramáticas cifras de desempleo entregadas por el Ministro de Trabajo, Angel Cabrera, anticipó que ya se analizan medidas de fondo como una eventual contratación por horas. ",
"label":"neutro"},
{"text":"Una ruana, un sombrero, manos ásperas por el trabajo, un machete al cinto, piel curtida por el sol y una gran y honesta sonrisa es la imagen de un campesino en mi memoria",
"label":"positivo"},
{"text":"Las cifras que siempre han dado son falsas, maquilladas, acomodados,  colombia es un país donde la mayoría de personas son independientes informales  .. no hay suficientes empresas que generen trabajo. La mayor empresa es el estado y esta manejada por solo corruptos",
"label":"negativo"},
{"text":"COLOMBIA no da garantías de tener un trabajo digno para pagar las cuotas.",
"label":"negativo"},
{"text":"Bajo ese planteamiento al parecer no importa que muchos se queden sin trabajo, lo importante es que quienes lo tiene mantengan los derechos laborales El desempleo en mayo en Colombia bordeará el 40%, pero al parecer no te importa.",
"label":"negativo"},
{"text":"España: Aprobado el INGRESO MÍNIMO VITAL para que quien no tenga salario logre cubrir las necesidades básicas. Colombia: Pensando en suspender la prima, reducir el salario mínimo, que regalemos dos horas de trabajo, no pagar los intereses de las cesantías, entre otras.",
"label":"negativo"},
{"text":"Tanto pregona de protección a los trabajadores y el distrito va a sacar y a dejar sin trabajo a más de 600 trabajadores del distrito, solo para meter gente como lo hacen en la secretaría de salud, contratos y liderazgos a gente que no tiene idea de nada.",
"label":"negativo"},
{"text":"Diego, el sena y las demás instituciones de educación superior, volverán a las aulas en agosto bajo un modelo de alternancia que combina clases presenciales y trabajo en casa",
"label":"neutro"},
{"text":"Se reconoce la labor del campesino, agricultor o ganadero quien hace producir, con su trabajo y esfuerzo, la tierra y los animales. Dios los bendiga por su gran labor.",
"label":"positivo"},
{"text":"En el proceso de reconstrucción económica y social de Colombia después del coronavirus, estamos en condiciones de convertirnos en un país digital,  con énfasis en el trabajo, la educación y en  producciones   y mercadeos urbanos  y rurales más justos.",
"label":"positivo"},
{"text":"Hay  dificultades de salud, economía, trabajo, movilidad y familia, pero todos podemos salir triunfadores, entre todos buscaremos soluciones. Los invito a que sigan soñando, que ninguno interrumpa sus estudios. Colombia los necesita.",
"label":"positivo"},
{"text":"Dios lo Bendiga Estimado presidente de la república de COLOMBIA, Su buen trabajo está dando frutos  Tenga la Fe Puesta en  Dios que todo Cambiara.",
"label":"positivo"},
{"text":"Conóce el módulo de Seguridad y Salud en el Trabajo; contempla las definiciones de la OIT y disposiciones legales de los Ministerios de Trabajo y normas reglamentarias.",
"label":"neutro"},
{"text":"En colpensiones dignificamos el trabajo del campo, acompañamos y asesoramos a los campesinos para que tengan una vejez tranquila como recompensa a su esfuerzo y a su contribución en la construcción de una Colombia Bandera de Colombia rica en biodiversidad.",
"label":"positivo"},
{"text":"Hemos destruido 20 años de trabajo de mejoras en las estadísticas sociales de Colombia, en apenas dos meses",
"label":"negativo"},
{"text":"Sin trabajo por la crisis desde fin de Marzo y el desempleo sigue y sigue en aumento no queda más que tener paciencia y seguir buscando saludo desde Colombia",
"label":"positivo"},
{"text":"En Colombia debemos establecer un nuevo enfoque de la política económica, a fin de promover el desarrollo empresarial con trabajo decente.",
"label":"neutro"},
{"text":"El Día de nuestros Campesinos, es una  oportunidad para reconocer el trabajo incansable de este importante actor no solo ahora en está crisis mundial sino en todos los tiempos.",
"label":"positivo"},
{"text":"Los trabajadores y trabajadoras del campo están sosteniendo la seguridad alimentaria del país. Hoy más que nunca el Gobierno debe brindarles condiciones de trabajo dignas.",
"label":"positivo"}
];
api_key ="tKqeiF4b8h6ElX2OfbXbVQKur"
api_secret ="PtXTNq45x2oieOtoJ0ixaotiYEsrkLmHyGKhKYfWqsclR4lYB7"
access_token ="1204934143534080000-OxFSPt0Q5HgroVtdOCp6OizVmMSrlb"
access_secret ="IAwOtyvrWt8WAQc9wlKfaRmdmA48exV54yCifZTgpLL6v"
auth = OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

places = api.geo_search(query="COL", granularity="country");
place_id = places[0].id;

listTweetText=[];
for tweet in tweepy.Cursor(api.search,q="racismo#place:%s#" % place_id,tweet_mode='extended',rpp=100,result_type="recent",lang="es",exclude_replies=True).items(100):
    print(tweet.full_text)
    listTweetText.append(tweet.full_text)
from pigeon import annotate 
from IPython.display import display,Image

annotations = annotate(listTweetText, options=['positivo', 'negativo','neutro'])
annotations
# CLASIFICADOR NAIVE BAYES 
from textblob.classifiers import NaiveBayesClassifier
with open('/kaggle/input/conjunto-entrenamiento/conjunto_entrenamiento.json','r') as fp:
    cl=NaiveBayesClassifier(fp, format="json");

print(cl.classify("Excelente trabajo . La política en Colombia es una empresa corrupta, donde se invierten unos millones en apoyar un candidato y luego se devuelven en contratos multimillonarios a los inversores, sus amigos y negocios con abogados y periodistas que cubren sus andanzas"));


places = api.geo_search(query="COL", granularity="country");
place_id = places[0].id;

listTweetText=[];
tema_teewt="coronavirus";
pais="place:%s#" % place_id;
for tweet in tweepy.Cursor(api.search,q=tema_teewt+"#"+pais,tweet_mode='extended',rpp=100,result_type="recent",lang="es",exclude_replies=True).items(5):
    print(limpiar_text(tweet.full_text));
    print(cl.classify(limpiar_text(tweet.full_text)));
    print("");

from textblob import TextBlob

places = api.geo_search(query="COL", granularity="country");
place_id = places[0].id;

listTweetText=[];
tema_teewt="coronavirus";
pais="place:%s#" % place_id;
for tweet in tweepy.Cursor(api.search,q=tema_teewt+"#"+pais,tweet_mode='extended',rpp=100,result_type="recent",lang="es",exclude_replies=True).items(5):
    print(limpiar_text(tweet.full_text));
    analysis = TextBlob(limpiar_text(tweet.full_text))
    print(analysis.sentiment);
    print("");
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


places = api.geo_search(query="COL", granularity="country");
place_id = places[0].id;

listTweetText=[];
tema_teewt="coronavirus";
pais="place:%s#" % place_id;
for tweet in tweepy.Cursor(api.search,q=tema_teewt+"#"+pais,tweet_mode='extended',rpp=100,result_type="recent",lang="es",exclude_replies=True).items(5):
    print(limpiar_text(tweet.full_text));
    vs = analyzer.polarity_scores(limpiar_text(tweet.full_text))
    print(vs)
    print("");


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

#docs = ['why hello there', 'omg hello pony', 'she went there? omg']
docs =[]
for fila_ in conjunto_entrenamiento:
    docs.append(fila_.get("text"))
vec = CountVectorizer()
X = vec.fit_transform(docs)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
print(df)