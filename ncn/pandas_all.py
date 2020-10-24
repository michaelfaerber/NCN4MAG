import pandas as pd


#this program creates the mag_all.txt file, which is zsed to perform machine learning on the sitation data of the mag.
#input: tsv mag dump:
path_PaperAuthorAffilion = "/pfs/work7/workspace/scratch/ucgvm-input-0/newmag/mag-2020-10-15/PaperAuthorAffiliations.txt"
path_Authors = "/pfs/work7/workspace/scratch/ucgvm-input-0/newmag/mag-2020-10-15/Authors.txt"
path_Papers = "/pfs/work7/workspace/scratch/ucgvm-input-0/newmag/mag-2020-10-15/Papers.txt"
path_PaperUrls = "/pfs/work7/workspace/scratch/ucgvm-input-0/newmag/mag-2020-10-15/PaperUrls.txt"
path_PaperFieldsofStudy = "/pfs/work7/workspace/scratch/ucgvm-input-0/newmag/mag-2020-10-15/PaperFieldsOfStudy.txt"
path_PaperCitationContexts = "/pfs/work7/workspace/scratch/ucgvm-input-0/newmag/mag-2020-10-15/PaperCitationContexts.txt"

#path where to write the compiled information
path_output_file = '/pfs/work7/workspace/scratch/ucgvm-input-0/input/mag_all_pandas_tsv.txt'
#inclusive bound
lower_bound_citationcount = 10

print("starting")

paperauthoraffiliations = pd.read_csv(path_PaperAuthorAffilion, sep="\t", names=["paperid", "authorid", "affiliationid", "authorsequencenumber", "originalauthor", "originalaffiliation"])
print("paperauthoraffiliations are loaded")

authors= pd.read_csv(path_Authors, sep="\t", names=["authorid", "rank", "normalizedname", "displayname", "lastknownaffiliationid", "papercount", "paperfamilydi", "citationcount", "createDate"])
print("authors are loaded")
#replace the authorid with authornames
papertoauthorname1 = pd.merge(paperauthoraffiliations, authors, left_on="authorid", right_on="authorid")[["paperid","displayname"]]
del authors
del paperauthoraffiliations

#aggregate the author names, eg: {paper1 - author jack;paper 1 - author john; paper 2 - author george} --> {paper1 - author jack, author john;paper 2 - author george}
papertoauthorname = papertoauthorname1.groupby("paperid").agg({'displayname': ",".join})
del papertoauthorname1


print("loading papers")
papers = pd.read_csv(path_Papers, sep="\t", names=["paperid", "rank", "doi", "doctype", "papertitle", "originaltitle", "booktitle", "year", "date", "onlinedate", "publisher", "journalid", "conferenceseriesid", "conferenceseriesinstanceid", "attribute name", "attribute name2", "volume", "issue", "firstpage", "lastpage", "referencecount", "citationcount", "estimatedcitation", "originalvenue", "familyid", "createddate", "paperid2", "indexedabstract"])
print("papers are loaded")

paperurls = pd.read_csv(path_PaperUrls, sep="\t", names=["paperid", "sourcetype", "sourceurl", "languagecode" ])
print("paperurls are loaded")

#inner join, to filter for english papers
print("step 0")
onlyenglishpapers=pd.merge(paperurls[paperurls.languagecode == "en"],papers, left_on="paperid", right_on="paperid")[["paperid", "year", "papertitle", "citationcount"]]
del paperurls
del papers

paperfieldsofstudy = pd.read_csv(path_PaperFieldsofStudy, sep="\t", names=["paperid", "fieldofstudyid", "score"])
print("fieldofstudy are loaded")

#inner join, to filter for comp. science  papers (41008148 is the id of fieldofstudy computer science
print("Step 1")
onlyenglishcs=pd.merge(paperfieldsofstudy[paperfieldsofstudy.fieldofstudyid == 41008148],onlyenglishpapers, left_on="paperid", right_on="paperid")[["paperid", "year", "papertitle", "citationcount"]]

del paperfieldsofstudy
del onlyenglishpapers

papercitationcontexts = pd.read_csv(path_PaperCitationContexts, sep="\t", names=["citingpaperid", "paperreferenceid", "citationcontext"])
print("citataioncontexts are loaded")

print("Step 2")
#papercitationscontexts is the data we want with metadata, add (cited) title here
#trying to fix the Type Error here:
papercitationcontexts.paperreferenceid.astype(int)
onlyenglishcs.paperid.astype(int)
onlyenglishcs.citataioncount.astype(int)
#filter for citationcount of cited paper & add cited title
contexts1=pd.merge(onlyenglishcs[onlyenglishcs.citationcount >= lower_bound_citationcount], papercitationcontexts, left_on="paperid", right_on="paperreferenceid")[["citingpaperid", "paperreferenceid", "citationcontext", "papertitle"]]
contexts1=contexts1.rename(columns={"papertitle":"citedtitle"})
del papercitationcontexts

#add citing title (as papertitle) and year of citing paper (as year)
contexts=pd.merge(onlyenglishcs, contexts1, left_on="paperid", right_on="citingpaperid")[["citingpaperid","year", "paperreferenceid", "citationcontext", "papertitle", "citedtitle"]]
del onlyenglishcs
del contexts1


print("Step 3")
#add cited authors
withcitedauthors = pd.merge(contexts, papertoauthorname, left_on="paperreferenceid", right_on="paperid")[["citingpaperid","year",  "papertitle","paperreferenceid", "citationcontext", "displayname", "citedtitle"]]
withcitedauthors = withcitedauthors.rename(columns={"displayname":"citedauthors"})

print("step 4")
#add citing authors
withallauthors = pd.merge(withcitedauthors, papertoauthorname, left_on="citingpaperid", right_on="paperid")[["citingpaperid","year", "papertitle","paperreferenceid", "citationcontext", "displayname","citedtitle", "citedauthors"]]
del withcitedauthors
del contexts
del papertoauthorname

withallauthors=withallauthors.rename(columns={"displayname":"citingauthors"})

print("remove duplicates")
#unique entries are defined by 1) citingpaper id, 2) citationcontext 3) paperreference id  - that way only real duplicates are eliminated (no idea where duplicates come from but there are many wihtout this line)
withallauthors=withallauthors.drop_duplicates(subset=["citingpaperid", "citationcontext", "paperreferenceid"])

print("Step 5: put out")

#final format:
#"citingpaperid","year", "papertitle","paperreferenceid", "citationcontext", "citingauthors","citedtitle", "citedauthors"


#tab seperated, because authors are comma seperated
with open(path_output_file, 'w') as f:
  withallauthors.to_csv(f, sep="\t", index=False)
print("done")


