import pandas as pd


#this program creates the mag_all.txt file, which is zsed to perform machine learning on the sitation data of the mag.
#input: tsv mag dump

print("starting")

paperauthoraffiliations = pd.read_csv("/pfs/work7/workspace/scratch/utdkf-mag-0/mag/PaperAuthorAffiliations.txt", sep="\t", names=["paperid", "authorid", "affiliationid", "authorsequencenumber", "originalauthor", "originalaffiliation"])
print("paperauthoraffiliations are loaded")

authors= pd.read_csv("/pfs/work7/workspace/scratch/utdkf-mag-0/mag/Authors.txt", sep="\t", names=["authorid", "rank", "normalizedname", "displayname", "lastknownaffiliationid", "papercount", "paperfamilydi", "citationcount", "createDate"])
print("authors are loaded")
#replace the authorid with authornames
papertoauthorname1 = pd.merge(paperauthoraffiliations, authors, left_on="authorid", right_on="authorid")[["paperid","displayname"]]
del authors
del paperauthoraffiliations

#aggregate the author names, eg: {paper1 - author jack;paper 1 - author john; paper 2 - author george} --> {paper1 - author jack, author john;paper 2 - author george}
papertoauthorname = papertoauthorname1.groupby("paperid").agg({'displayname': ",".join})
del papertoauthorname1


print("loading papers")
papers = pd.read_csv("/pfs/work7/workspace/scratch/utdkf-mag-0/mag/Papers.txt", sep="\t", names=["paperid", "rank", "doi", "doctype", "papertitle", "originaltitle", "booktitle", "year", "date", "onlinedate", "publisher", "journalid", "conferenceseriesid", "conferenceseriesinstanceid", "attribute name", "attribute name2", "volume", "issue", "firstpage", "lastpage", "referencecount", "citationcount", "estimatedcitation", "originalvenue", "familyid", "createddate", "paperid2", "indexedabstract"])
print("papers are loaded")

paperurls = pd.read_csv("/pfs/work7/workspace/scratch/utdkf-mag-0/mag/PaperUrls.txt", sep="\t", names=["paperid", "sourcetype", "sourceurl", "languagecode" ])
print("paperurls are loaded")

#inner join, to filter for english papers
print("step 0")
onlyenglishpapers=pd.merge(paperurls[paperurls.languagecode == "en"],papers, left_on="paperid", right_on="paperid")[["paperid", "year", "papertitle", "citationcount"]]
del paperurls
del papers

paperfieldsofstudy = pd.read_csv("/pfs/work7/workspace/scratch/utdkf-mag-0/advanced/PaperFieldsOfStudy.txt", sep="\t", names=["paperid", "fieldofstudyid", "score"])
print("fieldofstudy are loaded")

#inner join, to filter for comp. science  papers (41008148 is the id of fieldofstudy computer science
print("Step 1")
onlyenglishcs=pd.merge(paperfieldsofstudy[paperfieldsofstudy.fieldofstudyid == 41008148],onlyenglishpapers, left_on="paperid", right_on="paperid")[["paperid", "year", "papertitle", "citationcount"]]

del paperfieldsofstudy
del onlyenglishpapers

papercitationcontexts = pd.read_csv("/pfs/work7/workspace/scratch/utdkf-mag-0/nlp/PaperCitationContexts.txt", sep="\t", names=["citingpaperid", "paperreferenceid", "citationcontext"])
print("citataioncontexts are loaded")

print("Step 2")
#papercitationscontexts is the data we want with metadata, add (cited) title here
#trying to fix the Type Error here:
papercitationcontexts.paperreferenceid.astype(int)
onlyenglishcs.paperid.astype(int)
contexts1=pd.merge(onlyenglishcs, papercitationcontexts, left_on="paperid", right_on="paperreferenceid")[["citingpaperid", "paperreferenceid", "citationcontext", "papertitle"]]
contexts1=contexts1.rename(columns={"papertitle":"citedtitle"})
del papercitationcontexts

#add citing title (as papertitle) and year of citing paper (as year)
contexts=pd.merge(onlyenglishcs, contexts1, left_on="paperid", right_on="citingpaperid")[["citingpaperid","year", "paperreferenceid", "citationcontext", "papertitle", "citedtitle", "citationcount"]]
del onlyenglishcs
del contexts1


print("Step 3")
#add cited authors
withcitedauthors = pd.merge(contexts, papertoauthorname, left_on="paperreferenceid", right_on="paperid")[["citingpaperid","year",  "papertitle","paperreferenceid", "citationcontext", "displayname", "citedtitle", "citationcount"]]
withcitedauthors = withcitedauthors.rename(columns={"displayname":"citedauthors"})

print("step 4")
#add citing authors
withallauthors = pd.merge(withcitedauthors, papertoauthorname, left_on="citingpaperid", right_on="paperid")[["citingpaperid","year", "papertitle","paperreferenceid", "citationcontext", "displayname","citedtitle", "citedauthors", "citationcount"]]
del withcitedauthors
del contexts
del papertoauthorname

withallauthors=withallauthors.rename(columns={"displayname":"citingauthors"})

print("remove duplicates")
withallauthors=withallauthors.drop_duplicates(subset=["citingpaperid", "citationcontext", "paperreferenceid"])

print("Step 5: put out")

#final format:
#"citingpaperid","year", "papertitle","paperreferenceid", "citationcontext", "citingauthors","citedtitle", "citedauthors"


#tab seperated, because authors are comma seperated
with open('/pfs/work7/workspace/scratch/ucgvm-input-0/input/mag_all_pandas_tsv.txt', 'w') as f:
  withallauthors.to_csv(f, sep="\t", index=False)
print("done")


