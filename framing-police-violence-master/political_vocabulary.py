# The actors (subjects) of interest
ENTITIES = {
    # ethnicities & races
    'ethnicities_races': {
        'Israel': ['Israel', 'Israeli', 'Israelis'],
        'Maronite': ['Maronite', 'Maronites'],
        'Baathist': ['Baathist', 'Baathists'],
        'Shia': ['Shia', 'Shias', 'Shiites'],
        'Guardian': ['Gaurdian', 'Guardians'],
        'Phalangist': ['Phalangist', 'Phalangists', 'Phalange'],
        'Christian': ['Christian', 'Christians'],
        'Palestinian': ['Palestine', 'Palestinian', 'Palestinians'],
        'Soviet': ['Soviet', 'Soviets'],
        'Iraqi': ['Iraq', 'Iraqi', 'Iraqis'],
        'Communist': ['Communist', 'Communists'],
        'European': ['Europe', 'European', 'Europeans'],
        'Syrian': ['Syria', 'Syrian', 'Syrians'],
        'American': ['America', 'American', 'Americans', 'US', 'USA', 'United States', 'United States of America'],
        'Jewish': ['Jew', 'Jews', 'Jewish'],
        'Arabic': ['Arab', 'Arabs', 'Arabic'],
        'British': ['Britain', 'British', 'UK', 'United Kingdom'],
        'Lebanese': ['Lebanon', 'Lebanese'],
        'Jordanian': ['Jordan', 'Jordanian', 'Jordanians'],
        'Sunni': ['Sunni', 'Sunna'],
        'Saudi': ['Saudi', 'Saudis', 'Saudi Arabia', 'KSA', 'Kingdom of Saudi Arabia'],
        'Muslim': ['Muslim', 'Muslims'],
        'Iranian': ['Iran', 'Iranian', 'Iranians'],
        'Persian': ['Persian', 'Persians'],


        'Imperial': ['imperial', 'imperials', 'imperialist', 'imperialists']
    },

    # political parties
    'political_parties': [
        'Abu Nidal Organization', 'ANO', 'Fatah – The Revolutionary Council', 'al-Aqsa Martyrs Brigades', 'Amal Movement', 'Arab Liberation Army',
        'Arab Liberation Front', 'Jaysh al-Islam', 'Army of the Holy War', 'As-Sa\'iqa', 'Assaiqa', 'Black September Organization',
        'Democratic Front for the Liberation of Palestine', 'Fatah', 'Gaurdians of the Cedars', 'Hadash', 'Hamas',
        'Hezbollah', 'Hezbullah', 'Palestinian Islamic Jihad', 'Islamic Unification Movement', 'Japanese Red Army',
        'Kataeb', 'Kataeb Party', 'Phalanges', 'Lebanese Forces', 'Likud', 'Palestine Liberation Organization',
        'Palestinian Liberation Organization', 'PLO', 'Palestine Liberation Army', 'PLA', 'Palestinian Liberation Front',
        'PLF', 'Palestinian Popular Struggle Front', 'PPSF', 'Popular Front for the Liberation of Palestine',
        'Popular Front for the Liberation of Palestine - General Command', 'PFLP-GC', 'Popular Resistance Committees',
        'PRC', 'Progressive Socialist Party', 'Revolutionary Cells', 'RZ', 'Syrian Social Nationalist Party', 'SSNP'
    ],

    'politicians': ['Abbas al-Musawi', 'almusawi', 'Abu Nidal', 'Bachir Gemayel', 'Gemayel', 'Elie Hobeika', 'Hobeika', 'Hafez Al-Assad',
                   'Al-Assad', 'Imad Mughniyeh', 'Mughniyeh', 'Nabih Berri', 'Berri', 'Osama Bin Laden', 'Bin Laden',
                   'Rafik Hariri', 'Rafiq Hariri', 'Hariri', 'Ragheb Harb', 'Harb', 'Rashid Karami', 'Karami',
                   'Saad Haddad', 'Haddad', 'Yasser Arafat', 'Arafat'],

    'ideologies': [
        'Salafism', 'communism', 'National liberalism', 'Palestinianism','Armed struggle',
        'Social conservatism', 'Conservatism', 'Anti-West', 'Palestinian nationalism',
        'United Armenia', 'Anti-Zionism', 'Democratic socialism', 'Khomeinism', 'Islamism',
        'Socialism', 'Feminism', 'Arabism', 'West', 'Jihadism', 'Shia Jihad', 'Baathism',
        'Antisemitism', 'Anti-Syrian Government', 'Marxism', 'imperialism', 'Syrian Government',
        'Islamic Fundamentalism', 'Christian nationalism', 'Communism', 'Social democracy',
        'Lebanese nationalism', 'Anti-Arabism', 'Zionism', 'Arab nationalism',
        ' anti-Zionism', 'pan', 'Nasserism', 'Pan-Arabism', 'Anti-pan-Arabism',
        'National conservatism', 'Pan-Syrianism', 'Arab socialism', 'Anti-Palestinian',
        'Anti-imperialism', 'Islamic nationalism', 'Pan-Islamism', 'Secularism',
        'Economic liberalism', 'Phoenicianism', 'Marxism–Leninism', 'Anti-communism',
        'Armenian irredentism (United Armenia)', 'Sunni Islamism', 'Populism',
        'Liberal conservatism', 'Palestinian', 'Ultranationalism', 'Armenian nationalism',
    ]
}



NOUNS_ADJ = {

    # The ideologies as NOUNS
    'is.Imperialist': ['imperialism'],
    'is.Salafist': ['salafism'],
    'is.Communist': ['communism'],
    'is.Liberal': ['liberalism'],
    'is.Conservative': ['conservatism'],
    'is.Nationalist': ['nationalism'],
    'is.Zionist': ['zionism'],
    'is.democratic': ['democratism', 'democratic'],
    'is.socialist': ['socialism', 'socialist'],
    'is.Khomeini': ['khomeinism'],
    'is.Islamist': ['islamism'],
    'is.Feminist': ['feminism'],
    'is.Arabist': ['arabism'],
    'is.Jihadist': ['jihadism'],
    'is.Baathist': ['baathism'],

    # TBC

    ###################################################

    'is.aggressive': ['aggressive', 'aggression', 'aggressiveness'],
    'is.ally': ['ally', 'allies', 'agent', 'agents'],
    'is.attacked': ['attacked'],
    'is.assassination': ['assassination'],
    'is.assistive': ['assistance'],

    'is.brutal': ['brutal'],
    'is.barbaric': ['barbaric'],
    'is.blockade': ['blockade'],

    'is.committed': ['committed'],
    'is.criminal': ['criminal', 'criminality', 'cruel', 'cruelty', 'crime', 'crimes'],
    'is.controlling': ['control'],

    'is.damaged': ['damaged'],
    'is.destructive': ['destructive', 'destructing'],
    'is.defeated': ['defeat'],
    'is.dominating': ['domination'],
    'is.dangerous': ['dangerous'],
    'is.devastating': ['devastating'],
    'is.defending': ['defending'],

    'is.enemy': ['enemy', 'enemies'],
    'is.expelled': ['expelled'],

    'is.fighter': ['fighter', 'fighters', 'fighting'],
    'is.fail': ['failing', 'failure', 'failed'],
    'is.foolish': ['foolish', 'foolishness'],
    'is.fabricating': ['fabrication', 'fabrications'],

    'is.honorable': ['honorable'],
    'is.hideous': ['hideous'],
    'is.helpful': ['helpful', 'helping'],
    'is.hypocrite': ['hypocrite', 'hypocrisy'],
    'is.humiliating': ['humiliation'],

    'is.massacres': ['massacre', 'massacres'],
    'is.military': ['military'],
    'is.mistaken': ['mistake', 'miscalculation', 'error'],
    'is.murderer': ['murderer', 'murder'],

    'is.occupied': ['occupied'],
    'is.occupying': ['occupying', 'occupation'],

    'is.pressured': ['pressure', 'pressured'],

    'is.ready': ['ready', 'readiness'],

    'is.successful': ['success', 'achievement', 'achievements', 'victory', 'victories'],
    'is.struggling': ['struggle', 'struggles'],
    'is.suffering': ['suffering', 'suffer', 'suffers'],
    'is.supporting': ['support'],

    'is.terrorist': ['terrorism', 'terrorist'],
    'is.threatening': ['threatening', 'threat'],
    'is.terrible': ['terrible'],

    'is.unable': ['unable', 'inability'],
    'is.united': ['united', 'unity'],

    'is.valiant': ['valiant'],
    'is.victory': ['victory', 'victories'],
    'is.vulnerable': ['vulnerable'],

    'is.warhead': ['warhead'],
    'is.weak': ['weak', 'weakness'],
    'is.weakened': ['weakened', 'weakening']
}

VERBS = {
    'is.accusing': ['accuse', 'accuses', 'accusing', 'accused'],
    'is.assassinating': ['assassinate', 'assassinates', 'assassinating', 'assassinated'],
    'is.arming': ['arm', 'arms', 'arming', 'armed'],
    'is.attacking': ['attack', 'attacks', 'attacking', 'attacked'],
    'is.accomplished': ['accomplish', 'accomplishes', 'accomplishing', 'accomplished'],
    'is.assissting': ['assist', 'assists', 'assisting', 'assisted'],

    'is.commit': ['commit', 'commits', 'committing', 'committed'],
    'is.contributing': ['contribute', 'contributes', 'contributing', 'contributed'],
    'is.confronting': ['confront', 'confronts', 'confronting', 'confronted'],
    'is.controlling': ['control', 'controls', 'controlling', 'controlled'],
    'is.cooperating': ['cooperate', 'cooperates', 'cooperating', 'cooperated'],

    'is.damaging': ['damage', 'damages', 'damaging', 'damaged'],
    'is.disarming': ['disarm', 'disarms', 'disarming', 'disarmed'],
    'is.destroying': ['destroy', 'destroys', 'destroying', 'destroyed'],
    'is.destructing': ['destruct', 'destructs', 'destructing', 'destructed'],
    'is.defeating': ['defeat', 'defeats', 'defeating', 'defeated'],
    'is.dominating': ['dominate', 'dominates', 'dominating', 'dominated'],
    'is.defending': ['defend', 'defends', 'defending', 'defended'],

    'is.expelling': ['expel', 'expels', 'expelling', 'expelled'],

    'is.fighting': ['fights', 'fight', 'fighting', 'fought'],
    'is.fabricating': ['fabricate', 'fabricates', 'fabricating', 'fabricated'],
    'is.failing': ['fail', 'fails', 'failing', 'failed'],

    'is.helping': ['help', 'helps', 'helping', 'helped'],

    'is.intensifying': ['intensify', 'intensifies', 'intensifying', 'intensified'],

    'is.killing': ['kill', 'kills', 'killing', 'killed'],
    'is.kidnapping': ['kidnap', 'kidnaps', 'kidnapping', 'kidnapped'],

    'is.liberating': ['liberate', 'liberates', 'liberating', 'liberated'],

    'is.oppressed': ['oppressed'],
    'is.oppressing': ['oppress', 'oppresses', 'oppressing'],

    'is.pressuring': ['pressuring'],

    'is.struggling': ['struggle', 'struggles', 'struggling', 'struggled'],
    'is.striking': ['strike', 'strikes', 'striking', 'struck', 'struck'],
    'is.supporting': ['support', 'supports', 'supporting', 'supported'],
    'is.sacrificing': ['sacrifice', 'sacrifices', 'sacrificing', 'sacrificed'],
    'is.suffering': ['suffer', 'suffers', 'suffered', 'suffering'],

    'is.trapping': ['trap', 'traps', 'trapping', 'trapped'],
    'is.threatening': ['threat', 'threats', 'threaten', 'threatens', 'threatening', 'threatened'],

    'is.uniting': ['unite', 'unites', 'uniting', 'united'],

    'is.violating': ['violate', 'violates', 'violating', 'violated'],

    'is.weakening': ['weaken', 'weakens', 'weakening', 'weakened'],
}