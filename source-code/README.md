# Scrape single-page in Python Actor template

A template for scraping data from a single web page in Python. The URL of the web page is passed in via input, which is defined by the [input schema](https://docs.apify.com/platform/actors/development/input-schema). The template uses the [Requests](https://requests.readthedocs.io/) to get the HTML of the page and the [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) to parse the data from it. The data are then stored in a [dataset](https://docs.apify.com/sdk/js/docs/guides/result-storage#dataset) where you can easily access them.

The scraped data in this template are page headings but you can easily edit the code to scrape whatever you want from the page.

## Included features

- **[Apify SDK](https://docs.apify.com/sdk/js/)** - toolkit for building Actors
- **[Input schema](https://docs.apify.com/platform/actors/development/input-schema)** - define and easily validate a schema for your Actor's input
- **[Dataset](https://docs.apify.com/sdk/js/docs/guides/result-storage#dataset)** - store structured data where each object stored has the same attributes
- **[Requests](https://requests.readthedocs.io/)** - library for making HTTP requests in Python
- **[Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)** - library for pulling data out of HTML and XML files



## How it works

1. `Actor.get_input()` gets the input where the page URL is defined
2. `requests.get(url)` fetches the page
3. `BeautifulSoup(response.content, 'html.parser')` loads the page data and enables parsing the headings
4. This parses the headings from the page and here you can edit the code to parse whatever you need from the page

    ```python
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
    ```

5. `Actor.push_data(headings)` stores the headings in the dataset


## Getting started

For complete information [see this article](https://docs.apify.com/platform/actors/development#build-actor-at-apify-console). In short, you will:

1. Build the Actor
2. Run the Actor

## Pull the Actor for local development

If you would like to develop locally, you can pull the existing Actor from Apify console using Apify CLI:

1. Install `apify-cli`

    **Using Homebrew**

    ```
    brew install apify/tap/apify-cli
    ```

    **Using NPM**

    ```
    npm -g install apify-cli
    ```

2. Pull the Actor by its unique `<ActorId>`, which is one of the following:
    - unique name of the Actor to pull (e.g. "apify/hello-world")
    - or ID of the Actor to pull (e.g. "E2jjCZBezvAZnX8Rb")

    You can find both by clicking on the Actor title at the top of the page, which will open a modal containing both Actor unique name and Actor ID.

    This command will copy the Actor into the current directory on your local machine.

    ```
    apify pull <ActorId>
    ```

## Documentation reference

To learn more about Apify and Actors, take a look at the following resources:

- [Apify SDK for JavaScript documentation](https://docs.apify.com/sdk/js)
- [Apify SDK for Python documentation](https://docs.apify.com/sdk/python)
- [Apify Platform documentation](https://docs.apify.com/platform)
- [Join our developer community on Discord](https://discord.com/invite/jyEM2PRvMU)
