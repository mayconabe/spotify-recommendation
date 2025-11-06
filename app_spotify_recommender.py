import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title='🎧 Recomendador Spotify (Conteúdo + Híbrido)', page_icon='🎧', layout='wide')

PRIMARY_URL = 'https://raw.githubusercontent.com/sushmaakoju/spotify-tracks-data-analysis/main/SpotifyFeatures.csv'
SECONDARY_URL = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-01-21/spotify_songs.csv'

@st.cache_data(show_spinner=True)
def load_data():
    urls = [PRIMARY_URL, SECONDARY_URL]
    for u in urls:
        try:
            df = pd.read_csv(u)
            if df.shape[0] > 1000:
                return df, u
        except Exception:
            pass
    sample = pd.DataFrame([
        {'track_id':1,'track_name':'Blinding Lights','artists':'The Weeknd','album_name':'After Hours','popularity':95,'danceability':0.514,'energy':0.73,'loudness':-5.934,'speechiness':0.0583,'acousticness':0.00146,'instrumentalness':0.0,'liveness':0.0897,'valence':0.334,'tempo':171.005,'duration_ms':200000,'genres':'pop'},
        {'track_id':2,'track_name':'Levitating','artists':'Dua Lipa','album_name':'Future Nostalgia','popularity':92,'danceability':0.702,'energy':0.8,'loudness':-4.96,'speechiness':0.0605,'acousticness':0.0408,'instrumentalness':0.0,'liveness':0.0895,'valence':0.583,'tempo':103.497,'duration_ms':203000,'genres':'pop; dance pop'},
        {'track_id':3,'track_name':'Watermelon Sugar','artists':'Harry Styles','album_name':'Fine Line','popularity':88,'danceability':0.735,'energy':0.554,'loudness':-7.818,'speechiness':0.0463,'acousticness':0.0725,'instrumentalness':0.0,'liveness':0.108,'valence':0.665,'tempo':95.973,'duration_ms':174000,'genres':'pop'},
        {'track_id':4,'track_name':'Shape of You','artists':'Ed Sheeran','album_name':'Divide (Deluxe)','popularity':96,'danceability':0.825,'energy':0.652,'loudness':-3.183,'speechiness':0.0802,'acousticness':0.581,'instrumentalness':0.0,'liveness':0.0939,'valence':0.931,'tempo':95.977,'duration_ms':233000,'genres':'pop'},
        {'track_id':5,'track_name':'Bad Guy','artists':'Billie Eilish','album_name':'When We All Fall Asleep','popularity':90,'danceability':0.704,'energy':0.434,'loudness':-10.383,'speechiness':0.1,'acousticness':0.33,'instrumentalness':0.0,'liveness':0.12,'valence':0.562,'tempo':135.128,'duration_ms':194000,'genres':'electropop; pop'}
    ])
    return sample, 'embedded_sample'

def preprocess(df_raw):
    rename_map = {
        'name': 'track_name', 'artists': 'artists', 'artist_name': 'artists',
        'track_id': 'track_id', 'id': 'track_id',
        'popularity': 'popularity', 'danceability': 'danceability', 'energy': 'energy',
        'loudness': 'loudness', 'speechiness': 'speechiness', 'acousticness': 'acousticness',
        'instrumentalness': 'instrumentalness', 'liveness': 'liveness', 'valence': 'valence',
        'tempo': 'tempo', 'duration_ms': 'duration_ms', 'genre': 'genres', 'genres': 'genres',
        'album_name': 'album_name', 'album': 'album_name', 'track': 'track_name'
    }
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    essential = ['track_name','artists','danceability','energy','loudness','speechiness',
                 'acousticness','instrumentalness','liveness','valence','tempo','popularity','album_name']
    df = df.dropna(subset=['track_name']).copy()
    if 'artists' not in df.columns:
        df['artists'] = 'desconhecido'
    df['artists'] = df['artists'].fillna('desconhecido').astype(str)
    df['track_name'] = df['track_name'].astype(str)
    df['key_name'] = df['track_name'].str.strip() + ' — ' + df['artists'].str.strip()

    numeric_feats = [c for c in essential if c in df.columns and c not in ['track_name','artists','album_name','genres']]
    df_num = df[numeric_feats].select_dtypes(include=['number']).copy()

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df_num.values) if (df_num.shape[0] and df_num.shape[1]) else np.zeros((len(df),1))
    return df, df_num, X, numeric_feats

def recomendar_conteudo(df, X, key_name, k=7):
    idx_list = df.index[df['key_name'] == key_name].tolist()
    if not idx_list:
        return pd.DataFrame(columns=['track_name','artists','similaridade'])
    idx = idx_list[0]
    sim = cosine_similarity(X[idx].reshape(1, -1), X)[0]
    order = np.argsort(sim)[::-1]
    order = order[sim[order] < 0.9999]
    top_idx = order[:k]
    cols = ['track_name','artists']
    if 'album_name' in df.columns:
        cols.append('album_name')
    out = df.iloc[top_idx][cols].copy()
    out['similaridade'] = sim[top_idx]
    return out

def recomendar_hibrido(df, X, key_name, k=7, alpha=0.7):
    alpha = max(0.0, min(1.0, alpha))
    idx_list = df.index[df['key_name'] == key_name].tolist()
    if not idx_list:
        return pd.DataFrame(columns=['track_name','artists','score','content_sim','collab_proxy'])
    idx = idx_list[0]
    sim = cosine_similarity(X[idx].reshape(1, -1), X)[0]
    collab = np.zeros(len(df), dtype=float)
    if 'popularity' in df.columns:
        pop = pd.to_numeric(df['popularity'], errors='coerce').fillna(0.0).values
        pmin, pmax = np.nanmin(pop), np.nanmax(pop)
        denom = (pmax - pmin) if (pmax - pmin) != 0 else 1.0
        pop_norm = (pop - pmin) / denom
        collab += pop_norm
    if 'artists' in df.columns:
        ref_artist = df.loc[idx, 'artists']
        same_artist = (df['artists'] == ref_artist).astype(float).values
        collab += 0.15 * same_artist
    cmin, cmax = np.min(collab), np.max(collab)
    collab = (collab - cmin) / (cmax - cmin + 1e-9) if (cmax - cmin) != 0 else np.zeros_like(collab)
    score = alpha * sim + (1 - alpha) * collab
    order = np.argsort(score)[::-1]
    order = order[order != idx]
    top_idx = order[:k]
    cols = ['track_name','artists']
    if 'album_name' in df.columns:
        cols.append('album_name')
    out = df.iloc[top_idx][cols].copy()
    out['score'] = score[top_idx]
    out['content_sim'] = sim[top_idx]
    out['collab_proxy'] = collab[top_idx]
    return out.sort_values('score', ascending=False)

# ---- UI ----
st.title('🎧 Recomendador Spotify — Conteúdo + Híbrido')
df_raw, source_used = load_data()
st.caption(f'Fonte de dados: {source_used}')
df, df_num, X, feats = preprocess(df_raw)

with st.expander('📦 Amostra do dataset'):
    st.write('Shape:', df.shape)
    st.dataframe(df.head(15))

left, right = st.columns([2,1])
with left:
    st.subheader('🔎 Faixa de referência')
    search = st.text_input('Busque por nome ou artista', 'Shape of You')
    options = df[df['key_name'].str.contains(search, case=False, na=False)]['key_name'].head(100).tolist()
    if not options and len(df) > 0:
        options = df['key_name'].head(50).tolist()
    selected = st.selectbox('Resultados', options)

with right:
    st.subheader('⚙️ Parâmetros')
    k = st.slider('Top‑K', min_value=3, max_value=20, value=10, step=1)
    alpha = st.slider('α (peso do conteúdo no híbrido)', min_value=0.0, max_value=1.0, value=0.7, step=0.05)

tab1, tab2, tab3 = st.tabs(['Conteúdo', 'Híbrido', 'Comparativo'])

if st.button('🚀 Recomendar'):
    with tab1:
        st.subheader('Conteúdo')
        recs_c = recomendar_conteudo(df, X, selected, k=k)
        st.dataframe(recs_c.reset_index(drop=True))
        try:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.bar(range(len(recs_c)), recs_c['similaridade'].values)
            plt.xticks(range(len(recs_c)), recs_c['track_name'].values, rotation=45, ha='right')
            plt.title('Similaridade (cosseno) — Conteúdo')
            st.pyplot(fig)
        except Exception:
            st.info('Matplotlib indisponível.')

    with tab2:
        st.subheader('Híbrido')
        recs_h = recomendar_hibrido(df, X, selected, k=k, alpha=alpha)
        st.dataframe(recs_h.reset_index(drop=True))
        try:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.plot(range(len(recs_h)), recs_h['score'].values, marker='o')
            plt.xticks(range(len(recs_h)), recs_h['track_name'].values, rotation=45, ha='right')
            plt.title('Score — Híbrido')
            st.pyplot(fig)
        except Exception:
            st.info('Matplotlib indisponível.')

    with tab3:
        st.subheader('Comparativo')
        recs_c = recomendar_conteudo(df, X, selected, k=k).copy().reset_index(drop=True)
        recs_c['rank_conteudo'] = recs_c.index + 1
        recs_c.rename(columns={'similaridade':'score_conteudo'}, inplace=True)

        recs_h = recomendar_hibrido(df, X, selected, k=k, alpha=alpha).copy().reset_index(drop=True)
        recs_h['rank_hibrido'] = recs_h.index + 1
        recs_h.rename(columns={'score':'score_hibrido'}, inplace=True)

        def _key(dfx):
            return dfx['track_name'].astype(str).str.strip() + ' — ' + dfx['artists'].astype(str).str.strip()
        recs_c['key'] = _key(recs_c)
        recs_h['key'] = _key(recs_h)

        comp = recs_c.merge(recs_h, on='key', how='outer', suffixes=('_conteudo','_hibrido'))
        cols_show = ['key','rank_conteudo','score_conteudo','rank_hibrido','score_hibrido']
        extra = [c for c in ['album_name_conteudo','album_name_hibrido'] if c in comp.columns]
        cols_show += extra
        comp_view = comp[cols_show].sort_values(by=['rank_conteudo','rank_hibrido'], na_position='last').reset_index(drop=True)
        st.dataframe(comp_view)

with st.expander('ℹ️ Como interpretar'):
    st.markdown('''**Conteúdo:** similaridade do cosseno em atributos numéricos (dançabilidade, energia, valence, tempo etc.).  
**Híbrido:** combinação linear entre conteúdo (α) e um proxy coletivo (popularidade normalizada + bônus para mesmo artista).  
**Leitura dos scores:** conteúdo próximo de 1.0 indica perfil muito semelhante; híbrido pondera preferência coletiva.  
**Evolução:** pesos por feature, re-ranking para diversidade, e colaborativo real quando houver histórico.''')
