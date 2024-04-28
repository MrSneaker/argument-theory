import datetime
import os
from helpers import *
from flask import Flask, render_template, request, redirect, url_for
app = Flask(__name__)

@app.route("/")
def root():

    return render_template("index.html")

def parsePreferencesFromText(preferences):
    prefsplit = [line.replace(" ","").replace("\r","").split(",") for line in preferences.strip("\n").split("\n")]
    print(prefsplit)
    prefs = setPrefDict(prefsplit)
    return prefs

def parsePreferencesFromWeights(ruleset):
    prefs = {}
    for rule in ruleset:
        if rule.isDefeasible:
            print(rule.name.name, rule.weight)
            prefs[rule.name.name] = rule.weight
    return prefs

@app.route("/read_rules", methods=["POST"])
def read_rules():

    #         return render_template("error.html",error_message="An error has occured. Please control your input and try again.")

    
    electionPrinciple = request.form["electionPrinciple"]
    linkPrinciple = request.form["linkPrinciple"]

    print("Election Principle: " + electionPrinciple)
    print("Link Principle: " + linkPrinciple)
    
    # Generate ruleset
    ruleset = [line2rule(line.strip("\r").strip("\n")) for line in request.form["rules"].strip("\n").strip(" ").split("\n")]
    add_Contrapositions(ruleset)

    # Generate Arguments and Attacks
    arguments = generateArguments(ruleset)
    for a in arguments:
        print(str(a))

    ras, rasusable = generateRebutAttacks(arguments)
    uas, uasusable = generateUndercutAttacksV2(arguments)

    prefs = None
    parsedfromweights = None
    if "ruleWeightsCheck" in request.form.keys():
        parsedfromweights = True
        prefs = parsePreferencesFromWeights(ruleset)
    else:
        parsedfromweights = False
        prefs = parsePreferencesFromText(request.form["preferences"])

    print(prefs)

    print(len(uas))

    # Obtain Defeats
    ranking = rankArguments(arguments, prefs, 
                        electionPrinciple=electionPrinciple,
                        linkPrinciple=linkPrinciple,
                        print_steps=False)
    successful_defeats = generateSuccessfulDefeats(arguments, ranking, prefs)
    filteredAttacks = differenceDefeatAttack(arguments, ranking, prefs)
    
    burdenRank = burden_rank(arguments, successful_defeats)
    
    filtered_attacks_display =  ["|".join(map(str, attack)) for attack in filteredAttacks]
    # burdenRank_display = ["|".join(x) for x in burdenRank]
    plot_url = plotDefeatByDegree(successful_defeats)

    return render_template("results.html",
                           ruleset=ruleset,
                           arguments=arguments,
                           rebuts=["|".join(x) for x in ras.values.tolist()],
                           undercuts=["|".join(x) for x in uas.values.tolist()],
                           defeats=[f"{d[0]}|defeats|{d[1]}" for d in successful_defeats],
                           filteredAtt=filtered_attacks_display,
                           plot_url=plot_url,
                           burdenRank=burdenRank)

if __name__ == "__main__":
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    port = os.getenv("PORT")
    if(port == None):
        port = 8082
    else:
        port = int(port)
    app.run(host="127.0.0.1", port=port, debug=True)