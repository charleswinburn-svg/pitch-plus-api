import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { useTheme } from "./ThemeContext.jsx";
import {
  fetchSchedule, fetchPlayByPlay, fetchBoxscore, fetchAllPlayers, fetchWbcPlayers, fetchGameLog, fetchLeagueStatLeaders,
  extractPitcherData, extractBatterData, aggregateByPitchType,
  fetchSavantGameData, enrichWithSavant, fetchSavantBulkXwoba,
  PITCH_COLORS, PITCH_NAMES, GAME_TYPE_LABELS, getWbcFlag, scorePitchCode,
} from "./mlbApi.js";
import {
  MovementPlot, SprayChart, ZonePlot, StatBar, PitchTable, PitchTypeLegend,
  PitcherCountTool, HitterCountTool, LocationZonePanel,
  RollingVeloChart, PlatoonUsageBars, ReclassifyModal,
} from "./SummaryComponents.jsx";
import { getHeadshotUrl, getLogoUrl, TEAM_IDS, saveCardAsPng, norm, MLB_TEAM_PRIMARY, hexLuminance, SearchableSelect } from "./SharedComponents.jsx";

const SEASON_TYPES = [
  { id: "S", label: "Spring Training" },
  { id: "W", label: "WBC" },
  { id: "R", label: "Regular Season" },
  { id: "P", label: "Postseason" },
];

export default function Summaries({ season, initialSubTab = "pitcher_game" }) {
  const { theme: t } = useTheme();
  
  const [subTab, setSubTab] = useState(initialSubTab);
  
  // Update subTab when initialSubTab changes from parent (dropdown selection)
  useEffect(() => {
    setSubTab(initialSubTab);
  }, [initialSubTab]);
  
  const [seasonType, setSeasonType] = useState("S");
  const [level, setLevel] = useState("MLB"); // "MLB" or "AAA"
  const [players, setPlayers] = useState(null);
  const [loadingPlayers, setLoadingPlayers] = useState(false);
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const [games, setGames] = useState([]);
  const [loadingGames, setLoadingGames] = useState(false);
  const [selectedGame, setSelectedGame] = useState(null);
  const [pbpData, setPbpData] = useState(null);
  const [boxscoreData, setBoxscoreData] = useState(null);
  const [savantData, setSavantData] = useState(null);
  const [loadingPbp, setLoadingPbp] = useState(false);
  const [error, setError] = useState(null);
  const [seasonPbps, setSeasonPbps] = useState([]);
  const [seasonSavant, setSeasonSavant] = useState([]);
  const [loadingSeason, setLoadingSeason] = useState(false);
  const [seasonProgress, setSeasonProgress] = useState("");
  const [bulkXwoba, setBulkXwoba] = useState(null);
  const [prospectXwoba, setProspectXwoba] = useState(null);
  const [leagueAvgs, setLeagueAvgs] = useState(null);
  const [baselineAvgs, setBaselineAvgs] = useState(null);
  const [pitchPlus, setPitchPlus] = useState(null);
  const [leaderboardPlus, setLeaderboardPlus] = useState(null);

  // ── Reclassification (session-only) ────────────────────────────────────
  // Two parallel maps: one for single-pitch overrides, one for bulk overrides
  // by pitch_type. Both are applied during enrichedData derivation; the score
  // useEffect below picks up the changes automatically because enrichedData
  // is in its dep array.
  const [reclassifyMode, setReclassifyMode] = useState(false);
  // pitchOverrides: Map<key, newType> where key = `${gamePk}_${atBatIndex}_${pitchNumber}`
  const [pitchOverrides, setPitchOverrides] = useState(() => new Map());
  // typeOverrides: Map<originalType, newType> — bulk reclassification per pitch_type
  const [typeOverrides, setTypeOverrides] = useState(() => new Map());
  // Modal state — which pitch is open for reclassification
  const [reclassifyTarget, setReclassifyTarget] = useState(null);

  const pitchOverrideKey = (p) => `${p.gamePk ?? ""}_${p.atBatIndex ?? ""}_${p.pitchNumber ?? ""}`;

  // Load pre-computed league avgs (from league_avgs_pipeline.py) as baseline
  useEffect(() => {
    // Try current season first, fall back to previous year
    const tryLoad = (yr) => fetch(`/league_avgs_${yr}.json`).then(r => r.ok ? r.json() : null).catch(() => null);
    tryLoad(season).then(d => {
      if (d) { setBaselineAvgs(d); return; }
      tryLoad(parseInt(season) - 1).then(d2 => { if (d2) setBaselineAvgs(d2); });
    });
  }, [season]);

  const isAAA = level === "AAA" && seasonType === "R";

  // Merge: computed leagueAvgs (from loaded games) takes priority, baseline fills gaps
  // Don't use MLB baseline JSON for AAA — let CountTool/PitchTable use AAA hardcoded defaults
  const effectiveLeagueAvgs = useMemo(() => {
    if (leagueAvgs) return leagueAvgs;
    if (isAAA) return null; // AAA uses LEAGUE_AVG_AAA hardcoded in SummaryComponents
    if (!baselineAvgs?.pitch_types) return null;
    // Convert baseline JSON format to the leagueAvgs format used by CountTool/PitchTable
    const ptAvgs = {};
    for (const [pt, data] of Object.entries(baselineAvgs.pitch_types)) {
      ptAvgs[pt] = {
        all: {
          zone: data.zone_pct,
          whiff: data.whiff_pct,
          ev: null,
          swing: null,
        },
      };
    }
    return { byPitchType: ptAvgs, byGroup: {} };
  }, [leagueAvgs, baselineAvgs, isAAA]);

  const isPitcher = subTab.startsWith("pitcher");
  const isGame = subTab.endsWith("game");
  const isCounts = subTab.endsWith("counts");
  const currentSeason = parseInt(season) || 2026;
  const sportId = isAAA ? 11 : 1;

  // Fetch bulk Savant xwOBA for the season (RS/PS/AAA — Savant has no expected stats for ST/WBC)
  useEffect(() => {
    setBulkXwoba(null);
    if (seasonType !== "R" && seasonType !== "P") return;
    const type = isPitcher ? "pitcher" : "batter";
    fetchSavantBulkXwoba(currentSeason, seasonType, type, isAAA)
      .then(data => setBulkXwoba(data))
      .catch(() => {});
  }, [currentSeason, seasonType, isPitcher, isAAA]);

  // Load prospect xwOBA for AAA
  useEffect(() => {
    if (!isAAA) { setProspectXwoba(null); return; }
    fetch("/prospect_xwoba.json")
      .then(r => r.json())
      .then(d => { setProspectXwoba(d); console.log(`[Prospect xwOBA] Loaded ${Object.keys(d).length} players`); })
      .catch(() => setProspectXwoba(null));
  }, [isAAA]);

  // Load players: start with roster, then supplement from game boxscores AND league stats
  useEffect(() => {
    setLoadingPlayers(true);
    setError(null);
    console.log(`[Summaries] Loading players: seasonType=${seasonType}, level=${level}`);
    if (seasonType === "W") {
      fetchWbcPlayers(currentSeason)
        .then(p => { setPlayers(p); setLoadingPlayers(false); })
        .catch(e => { setError(e.message); setLoadingPlayers(false); });
      return;
    }
    // Fetch roster + MLB stats leaders in parallel. The stats endpoint
    // returns EVERYONE who threw a pitch / had a PA in the league this season,
    // including DFA'd, released, AAA-to-MLB guys. This is the source of truth.
    Promise.all([
      fetchAllPlayers(currentSeason, sportId),
      fetchLeagueStatLeaders(currentSeason, seasonType, sportId),
    ])
      .then(([roster, leaders]) => {
        // Per-group dedup so two-way players (Ohtani etc.) appear in both lists.
        const seenPitchers = new Set(roster.pitchers.map(x => x.id));
        const seenHitters = new Set(roster.hitters.map(x => x.id));
        for (const p of leaders.pitchers) if (!seenPitchers.has(p.id)) { roster.pitchers.push(p); seenPitchers.add(p.id); }
        for (const p of leaders.hitters)  if (!seenHitters.has(p.id))  { roster.hitters.push(p);  seenHitters.add(p.id); }
        roster.pitchers.sort((a, b) => a.name.localeCompare(b.name));
        roster.hitters.sort((a, b) => a.name.localeCompare(b.name));
        console.log(`[Summaries] Loaded ${roster.pitchers.length} pitchers, ${roster.hitters.length} hitters`);
        setPlayers(roster);
        setLoadingPlayers(false);
      })
      .catch(e => { setError(e.message); setLoadingPlayers(false); });
  }, [currentSeason, seasonType, sportId]);

  // After games load, scan boxscores to find players not on the roster
  const hasScannedBoxscores = useRef(false);
  useEffect(() => {
    if (!games.length || !players || seasonType === "W" || hasScannedBoxscores.current) return;
    hasScannedBoxscores.current = true;
    // Per-group sets so a two-way player who's already a hitter can still be added as a pitcher
    const seenPitchers = new Set(players.pitchers.map(p => p.id));
    const seenHitters = new Set(players.hitters.map(p => p.id));
    let cancelled = false;
    (async () => {
      const newPitchers = [], newHitters = [];
      const BATCH = 8;
      const allGames = games;
      for (let i = 0; i < allGames.length; i += BATCH) {
        if (cancelled) break;
        const batch = allGames.slice(i, i + BATCH);
        const boxes = await Promise.all(
          batch.map(g => fetchBoxscore(g.gamePk).catch(() => null).then(b => ({ box: b, game: g })))
        );
        for (const { box, game } of boxes) {
          if (!box) continue;
          for (const side of ["away", "home"]) {
            const team = box.teams?.[side];
            if (!team) continue;
            const gameTeam = side === "away" ? game.away : game.home;
            const teamAbbr = gameTeam?.abbreviation || team.team?.abbreviation || "";
            const teamId = gameTeam?.id || team.team?.id;
            for (const [, p] of Object.entries(team.players || {})) {
              const person = p.person;
              if (!person) continue;
              const hasPitching = p.stats?.pitching && Object.keys(p.stats.pitching).length > 0;
              const hasBatting = p.stats?.batting && Object.keys(p.stats.batting).length > 0;
              const entry = { id: person.id, name: person.fullName, team: teamAbbr, teamId, position: p.position?.abbreviation };
              if (hasPitching && !seenPitchers.has(person.id)) {
                newPitchers.push(entry); seenPitchers.add(person.id);
              }
              if (hasBatting && !seenHitters.has(person.id)) {
                newHitters.push(entry); seenHitters.add(person.id);
              }
              if (!hasPitching && !hasBatting) {
                if (p.position?.type === "Pitcher" && !seenPitchers.has(person.id)) {
                  newPitchers.push(entry); seenPitchers.add(person.id);
                } else if (p.position?.type !== "Pitcher" && !seenHitters.has(person.id)) {
                  newHitters.push(entry); seenHitters.add(person.id);
                }
              }
            }
          }
        }
      }
      if (!cancelled && (newPitchers.length || newHitters.length)) {
        console.log(`[Summaries] Boxscore scan found ${newPitchers.length} additional pitchers, ${newHitters.length} hitters`);
        setPlayers(prev => ({
          pitchers: [...(prev?.pitchers || []), ...newPitchers].sort((a, b) => a.name.localeCompare(b.name)),
          hitters: [...(prev?.hitters || []), ...newHitters].sort((a, b) => a.name.localeCompare(b.name)),
        }));
      }
    })();
    return () => { cancelled = true; };
  }, [games, players, seasonType]);

  useEffect(() => {
    setLoadingGames(true);
    fetchSchedule(currentSeason, seasonType, isAAA ? 11 : null)
      .then(g => {
        const playable = g.filter(x => {
          const s = (x.status || "").toLowerCase();
          if (s.includes("scheduled") || s.includes("preview") || s.includes("postponed") || s.includes("cancelled") || s.includes("suspended")) return false;
          return true;
        });
        setGames(playable.reverse());
        setLoadingGames(false);
      })
      .catch(e => { setError(e.message); setLoadingGames(false); });
  }, [currentSeason, seasonType, isAAA]);

  useEffect(() => {
    if (!selectedGame || !isGame) return;
    setLoadingPbp(true);
    setPbpData(null);
    setBoxscoreData(null);
    setSavantData(null);
    setSelectedPlayer(null);
    Promise.all([
      fetchPlayByPlay(selectedGame.gamePk),
      fetchBoxscore(selectedGame.gamePk),
      fetchSavantGameData(selectedGame.gamePk).catch(() => null),
    ])
      .then(([pbp, box, sav]) => { setPbpData(pbp); setBoxscoreData(box); setSavantData(sav); setLoadingPbp(false); })
      .catch(e => { setError(e.message); setLoadingPbp(false); });
  }, [selectedGame, isGame]);

  const gamePlayers = useMemo(() => {
    if (!boxscoreData || !isGame) return [];
    const result = [];
    for (const side of ["away", "home"]) {
      const team = boxscoreData.teams?.[side];
      if (!team) continue;
      // Use remapped team from game schedule (handles Spring Breakout CRP→CIN)
      const gameTeam = selectedGame ? (side === "away" ? selectedGame.away : selectedGame.home) : null;
      const teamName = gameTeam?.abbreviation || team.team?.abbreviation || "";
      const teamId = gameTeam?.id || team.team?.id;
      for (const [key, p] of Object.entries(team.players || {})) {
        const person = p.person;
        if (!person) continue;
        const stats = p.stats || {};
        const isPitcherInGame = stats.pitching && Object.keys(stats.pitching).length > 0;
        const isHitterInGame = stats.batting && Object.keys(stats.batting).length > 0;
        if (isPitcher && isPitcherInGame) {
          result.push({ id: person.id, name: person.fullName, team: teamName, teamId, ip: stats.pitching.inningsPitched });
        } else if (!isPitcher && isHitterInGame) {
          result.push({ id: person.id, name: person.fullName, team: teamName, teamId, abs: stats.batting.atBats, hits: stats.batting.hits });
        }
      }
    }
    result.sort((a, b) => a.name.localeCompare(b.name));
    return result;
  }, [boxscoreData, isPitcher, isGame, selectedGame]);

  const playerList = isPitcher ? (players?.pitchers || []) : (players?.hitters || []);

  const [playerGamePks, setPlayerGamePks] = useState(null);

  // Fetch game log when player changes (for season view)
  // Skip for WBC — game log API doesn't cover sportId=51
  useEffect(() => {
    if (!selectedPlayer || isGame || seasonType === "W") { setPlayerGamePks(null); return; }
    const group = isPitcher ? "pitching" : "hitting";
    // sportId=1 (MLB) on MLB tab, sportId=11 (AAA) on AAA tab — keeps stats level-correct for two-way players
    fetchGameLog(selectedPlayer.id, currentSeason, group, sportId)
      .then(splits => {
        const pks = new Set(splits.map(s => s.game?.gamePk).filter(Boolean));
        setPlayerGamePks(pks.size > 0 ? pks : null);
      })
      .catch(() => setPlayerGamePks(null));
  }, [selectedPlayer, currentSeason, isPitcher, isGame, seasonType, sportId]);

  const playerGames = useMemo(() => {
    if (!selectedPlayer || !games.length || isGame) return [];
    if (playerGamePks && playerGamePks.size > 0) {
      return games.filter(g => playerGamePks.has(g.gamePk) || (g.isExhibition && seasonType === "S"));
    }
    // WBC: include ALL tournament games — extract functions filter by player ID
    if (seasonType === "W") {
      return games.filter(g => !g.isExhibition);
    }
    // AAA: team-based filter (player.teamId should match AAA team)
    if (isAAA) {
      return games.filter(g =>
        g.away?.id === selectedPlayer.teamId ||
        g.home?.id === selectedPlayer.teamId
      );
    }
    // Fallback to team-based filter
    return games.filter(g =>
      g.away?.id === selectedPlayer.teamId ||
      g.home?.id === selectedPlayer.teamId ||
      (g.isExhibition && seasonType === "S")
    );
  }, [selectedPlayer, games, isGame, playerGamePks, seasonType, isAAA]);

  const loadSeasonData = useCallback(async () => {
    if (!selectedPlayer || !playerGames.length || isGame) return;
    setLoadingSeason(true);
    setSeasonPbps([]);
    setSeasonSavant([]);
    const results = [];
    const savResults = [];
    for (let i = 0; i < playerGames.length; i++) {
      setSeasonProgress(`Loading game ${i + 1}/${playerGames.length}...`);
      try {
        const [pbp, box] = await Promise.all([
          fetchPlayByPlay(playerGames[i].gamePk),
          fetchBoxscore(playerGames[i].gamePk).catch(() => null),
        ]);
        results.push({ game: playerGames[i], pbp, box });
        const sav = await fetchSavantGameData(playerGames[i].gamePk).catch(() => null);
        if (sav) savResults.push(...sav);
      } catch (e) { /* skip */ }
    }
    setSeasonPbps(results);
    setSeasonSavant(savResults);
    setLoadingSeason(false);
    setSeasonProgress("");
  }, [selectedPlayer, playerGames, isGame]);

  useEffect(() => {
    if (!isGame && selectedPlayer && playerGames.length > 0) loadSeasonData();
  }, [isGame, selectedPlayer, playerGames.length]);

  // Compute league-wide averages from ALL pitches in loaded games (not just selected player)
  useEffect(() => {
    if (!seasonPbps.length) { setLeagueAvgs(null); return; }
    const COUNT_BUCKETS = [
      { id: "first", test: (b, s) => b === 0 && s === 0 },
      { id: "ahead", test: (b, s) => (b===0&&s===1)||(b===0&&s===2)||(b===1&&s===2) },
      { id: "even",  test: (b, s) => (b===1&&s===1)||(b===2&&s===2) },
      { id: "behind",test: (b, s) => (b===1&&s===0)||(b===2&&s===0)||(b===2&&s===1)||(b===3&&s===1)||(b===3&&s===0) },
      { id: "full",  test: (b, s) => b === 3 && s === 2 },
    ];
    const GROUPS = { Fastball: ["FF","SI","FC","FA"], Breaking: ["SL","CU","KC","CS","SV","ST","KN"], Offspeed: ["CH","FS","SC","FO"] };
    const getGroup = c => { for (const [g, codes] of Object.entries(GROUPS)) if (codes.includes(c)) return g; return "Other"; };
    const inZone = (pX, pZ, szT, szB) => Math.abs(pX) <= 0.88 && pZ >= (szB||1.5) && pZ <= (szT||3.5);

    // Extract ALL pitches from ALL plays in ALL loaded games
    const allPitches = [];
    for (const { pbp } of seasonPbps) {
      for (const play of (pbp.allPlays || [])) {
        let prevB = 0, prevS = 0;
        for (const evt of (play.playEvents || [])) {
          if (evt.isPitch && evt.pitchData) {
            allPitches.push({
              pt: evt.details?.type?.code || "UN",
              b: prevB, s: prevS,
              pX: evt.pitchData.coordinates?.pX,
              pZ: evt.pitchData.coordinates?.pZ,
              szT: evt.pitchData.strikeZoneTop,
              szB: evt.pitchData.strikeZoneBottom,
              isSwing: ["S","D","E","F","L","M","O","T","W","X"].includes(evt.details?.call?.code),
              isWhiff: ["S","W","T"].includes(evt.details?.call?.code),
              ev: evt.hitData?.launchSpeed || null,
              isInPlay: evt.details?.isInPlay,
            });
            if (evt.count) { prevB = evt.count.balls ?? prevB; prevS = evt.count.strikes ?? prevS; }
          }
        }
      }
    }

    if (!allPitches.length) { setLeagueAvgs(null); return; }
    console.log(`[League Avgs] Computing from ${allPitches.length} pitches across ${seasonPbps.length} games`);

    const bucketIds = ["all", ...COUNT_BUCKETS.map(b => b.id)];
    const getBucket = (b, s) => { for (const bk of COUNT_BUCKETS) if (bk.test(b, s)) return bk.id; return null; };

    // Compute per pitch type
    const ptAvgs = {};
    const pitchTypes = [...new Set(allPitches.map(p => p.pt))].filter(t => t !== "UN");
    for (const pt of pitchTypes) {
      ptAvgs[pt] = {};
      for (const bk of bucketIds) {
        const pool = bk === "all"
          ? allPitches.filter(p => p.pt === pt)
          : allPitches.filter(p => p.pt === pt && getBucket(p.b, p.s) === bk);
        const swings = pool.filter(p => p.isSwing);
        const whiffs = pool.filter(p => p.isWhiff);
        const zoned = pool.filter(p => p.pX != null && p.pZ != null);
        const inZ = zoned.filter(p => inZone(p.pX, p.pZ, p.szT, p.szB));
        const evs = pool.filter(p => p.isInPlay && p.ev).map(p => p.ev);
        ptAvgs[pt][bk] = {
          zone: zoned.length > 0 ? inZ.length / zoned.length * 100 : null,
          whiff: swings.length > 0 ? whiffs.length / swings.length * 100 : null,
          ev: evs.length > 0 ? evs.reduce((a, b) => a + b, 0) / evs.length : null,
          swing: pool.length > 0 ? swings.length / pool.length * 100 : null,
        };
      }
    }

    // Compute per pitch group
    const grpAvgs = {};
    for (const g of ["Fastball", "Breaking", "Offspeed"]) {
      grpAvgs[g] = {};
      for (const bk of bucketIds) {
        const pool = bk === "all"
          ? allPitches.filter(p => getGroup(p.pt) === g)
          : allPitches.filter(p => getGroup(p.pt) === g && getBucket(p.b, p.s) === bk);
        const swings = pool.filter(p => p.isSwing);
        const whiffs = pool.filter(p => p.isWhiff);
        const evs = pool.filter(p => p.isInPlay && p.ev).map(p => p.ev);
        grpAvgs[g][bk] = {
          swing: pool.length > 0 ? swings.length / pool.length * 100 : null,
          whiff: swings.length > 0 ? whiffs.length / swings.length * 100 : null,
          ev: evs.length > 0 ? evs.reduce((a, b) => a + b, 0) / evs.length : null,
        };
      }
    }

    setLeagueAvgs({ byPitchType: ptAvgs, byGroup: grpAvgs });
  }, [seasonPbps]);

  useEffect(() => {
    setSelectedPlayer(null);
    setSelectedGame(null);
    setPbpData(null);
    setBoxscoreData(null);
    setSavantData(null);
    setSeasonPbps([]);
    setSeasonSavant([]);
    hasScannedBoxscores.current = false;
  }, [subTab, seasonType, level]);

  const extractedData = useMemo(() => {
    if (isGame) {
      if (!pbpData || !selectedPlayer) return null;
      return isPitcher ? extractPitcherData(pbpData, selectedPlayer.id) : extractBatterData(pbpData, selectedPlayer.id);
    } else {
      if (!seasonPbps.length || !selectedPlayer) return null;
      if (isPitcher) {
        const allPitches = [];
        let totalOuts = 0, totalH = 0, totalR = 0, totalER = 0, totalK = 0, totalBB = 0, totalHBP = 0, totalHR = 0, totalBF = 0;
        for (const { pbp, box, game } of seasonPbps) {
          // Pitches from PBP (for charts, pitch table) — tag with game metadata
          // for season-level rolling charts that need to group by outing.
          const d = extractPitcherData(pbp, selectedPlayer.id);
          for (const p of d.pitches) allPitches.push({ ...p, gamePk: game?.gamePk, gameDate: game?.date });

          // Counting stats from boxscore (official MLB source)
          let usedBox = false;
          if (box) {
            for (const side of ["away", "home"]) {
              const ps = box.teams?.[side]?.players?.[`ID${selectedPlayer.id}`]?.stats?.pitching;
              if (ps && ps.inningsPitched != null) {
                const ipStr = String(ps.inningsPitched);
                const ipParts = ipStr.split(".");
                totalOuts += parseInt(ipParts[0]) * 3 + (parseInt(ipParts[1]) || 0);
                totalH += parseInt(ps.hits) || 0;
                totalR += parseInt(ps.runs) || 0;
                totalER += parseInt(ps.earnedRuns) || 0;
                totalK += parseInt(ps.strikeOuts) || 0;
                totalBB += parseInt(ps.baseOnBalls) || 0;
                totalHBP += parseInt(ps.hitBatsmen) || parseInt(ps.hitByPitch) || 0;
                totalHR += parseInt(ps.homeRuns) || 0;
                totalBF += parseInt(ps.battersFaced) || 0;
                usedBox = true;
                break;
              }
            }
          }
          // Fallback to PBP if boxscore missing
          if (!usedBox) {
            totalOuts += Math.floor(d.ip) * 3 + Math.round((d.ip % 1) * 10);
            totalH += d.hits; totalR += d.runs; totalER += d.ers;
            totalK += d.ks; totalBB += d.bbs; totalHBP += (d.hbps || 0); totalHR += (d.hrs || 0);
            totalBF += (d.bf || Math.floor(d.ip) * 3 + Math.round((d.ip % 1) * 10) + d.hits + d.bbs + (d.hbps || 0));
          }
        }
        const totalIP = Math.floor(totalOuts / 3) + (totalOuts % 3) / 10;
        if (totalBF === 0) totalBF = totalOuts + totalH + totalBB + totalHBP;
        return { pitches: allPitches, ip: totalIP, hits: totalH, runs: totalR, ers: totalER, ks: totalK, bbs: totalBB, hbps: totalHBP, hrs: totalHR, bf: totalBF, totalPitches: allPitches.length };
      } else {
        const allPitches = [], allAtBats = [], allBB = [];
        let totalPA = 0, boxH = 0, boxAB = 0, boxBB = 0, boxHBP = 0, boxSF = 0, boxTB = 0, usedBoxCount = 0;
        for (const { pbp, box } of seasonPbps) {
          const d = extractBatterData(pbp, selectedPlayer.id);
          allPitches.push(...d.pitches);
          allAtBats.push(...d.atBats); allBB.push(...d.battedBalls);

          // OBP/SLG from boxscore (official MLB source)
          let usedBox = false;
          if (box) {
            for (const side of ["away", "home"]) {
              const bs = box.teams?.[side]?.players?.[`ID${selectedPlayer.id}`]?.stats?.batting;
              if (bs && bs.plateAppearances != null) {
                totalPA += parseInt(bs.plateAppearances) || 0;
                boxH += parseInt(bs.hits) || 0;
                boxAB += parseInt(bs.atBats) || 0;
                boxBB += parseInt(bs.baseOnBalls) || 0;
                boxHBP += parseInt(bs.hitByPitch) || 0;
                boxSF += parseInt(bs.sacFlies) || 0;
                // TB = H + 2B + 2*3B + 3*HR
                const doubles = parseInt(bs.doubles) || 0;
                const triples = parseInt(bs.triples) || 0;
                const hrs = parseInt(bs.homeRuns) || 0;
                const singles = (parseInt(bs.hits) || 0) - doubles - triples - hrs;
                boxTB += singles + doubles * 2 + triples * 3 + hrs * 4;
                usedBox = true;
                usedBoxCount++;
                break;
              }
            }
          }
          if (!usedBox) {
            totalPA += d.pas;
          }
        }

        const evs = allBB.map(b => b.hitData?.launchSpeed).filter(Boolean);
        const avgEV = evs.length ? evs.reduce((a, b) => a + b, 0) / evs.length : null;
        const maxEV = evs.length ? Math.max(...evs) : null;
        const totalSwings = allPitches.filter(p => p.isSwing).length;
        const whiffs = allPitches.filter(p => p.isWhiff).length;
        const outside = allPitches.filter(p => p.pX != null && p.pZ != null && (Math.abs(p.pX) > 0.88 || p.pZ > (p.szTop || 3.5) || p.pZ < (p.szBot || 1.5)));
        const chased = outside.filter(p => p.isSwing).length;

        let obp, slg, kPct, bbPct;
        if (usedBoxCount > 0) {
          // Use boxscore totals for OBP/SLG
          const obpDenom = boxAB + boxBB + boxHBP + boxSF;
          obp = obpDenom > 0 ? Math.round((boxH + boxBB + boxHBP) / obpDenom * 1000) / 1000 : 0;
          slg = boxAB > 0 ? Math.round(boxTB / boxAB * 1000) / 1000 : 0;
          kPct = totalPA > 0 ? Math.round((allAtBats.filter(ab => ab.result?.event?.includes("Strikeout")).length) / totalPA * 1000) / 10 : 0;
          bbPct = totalPA > 0 ? Math.round(boxBB / totalPA * 1000) / 10 : 0;
        } else {
          // Fallback to PBP
          const ks = allAtBats.filter(ab => ab.result?.event?.includes("Strikeout")).length;
          const bbs = allAtBats.filter(ab => ["Walk","Intent Walk"].includes(ab.result?.event)).length;
          const hbps = allAtBats.filter(ab => ab.result?.event === "Hit By Pitch").length;
          const sfs = allAtBats.filter(ab => ["Sac Fly","Sacrifice Fly","Sac Fly DP"].includes(ab.result?.event)).length;
          const h = allAtBats.filter(ab => ["Single","Double","Triple","Home Run"].includes(ab.result?.event)).length;
          const abCount = allAtBats.filter(ab => !["Walk","Intent Walk","Hit By Pitch","Sacrifice Fly","Sacrifice Bunt","Sac Fly","Sac Bunt","Sac Fly DP","Catcher Interference","Batter Interference"].includes(ab.result?.event)).length;
          const tb = allAtBats.reduce((s, ab) => { const e = ab.result?.event; if (e === "Single") return s+1; if (e === "Double") return s+2; if (e === "Triple") return s+3; if (e === "Home Run") return s+4; return s; }, 0);
          const obpDenom = abCount + bbs + hbps + sfs;
          obp = obpDenom > 0 ? Math.round((h + bbs + hbps) / obpDenom * 1000) / 1000 : 0;
          slg = abCount > 0 ? Math.round(tb / abCount * 1000) / 1000 : 0;
          kPct = totalPA > 0 ? Math.round(ks / totalPA * 1000) / 10 : 0;
          bbPct = totalPA > 0 ? Math.round(bbs / totalPA * 1000) / 10 : 0;
        }

        const xwobaApprox = null;
        return { pitches: allPitches, pas: totalPA, obp, slg, avgEV: avgEV ? Math.round(avgEV*10)/10 : null, maxEV: maxEV ? Math.round(maxEV*10)/10 : null, kPct, bbPct, chasePct: outside.length > 0 ? Math.round(chased/outside.length*1000)/10 : 0, whiffPct: totalSwings > 0 ? Math.round(whiffs/totalSwings*1000)/10 : 0, xwoba: xwobaApprox, atBats: allAtBats, battedBalls: allBB };
      }
    }
  }, [isGame, isPitcher, pbpData, seasonPbps, selectedPlayer]);

  // Enrich with Savant xwOBA and VAA
  const enrichedData = useMemo(() => {
    if (!extractedData || !selectedPlayer) return extractedData;

    const result = { ...extractedData };

    // ER from boxscore for game view — PBP earned flags are unreliable
    if (isGame && isPitcher && boxscoreData) {
      for (const side of ["away", "home"]) {
        const pData = boxscoreData.teams?.[side]?.players?.[`ID${selectedPlayer.id}`];
        if (pData?.stats?.pitching?.earnedRuns != null) {
          result.ers = parseInt(pData.stats.pitching.earnedRuns) || 0;
          break;
        }
      }
    }

    // OBP/SLG from boxscore for game view hitters
    if (isGame && !isPitcher && boxscoreData) {
      for (const side of ["away", "home"]) {
        const bs = boxscoreData.teams?.[side]?.players?.[`ID${selectedPlayer.id}`]?.stats?.batting;
        if (bs && bs.plateAppearances != null) {
          const h = parseInt(bs.hits) || 0;
          const ab = parseInt(bs.atBats) || 0;
          const bb = parseInt(bs.baseOnBalls) || 0;
          const hbp = parseInt(bs.hitByPitch) || 0;
          const sf = parseInt(bs.sacFlies) || 0;
          const doubles = parseInt(bs.doubles) || 0;
          const triples = parseInt(bs.triples) || 0;
          const hrs = parseInt(bs.homeRuns) || 0;
          const singles = h - doubles - triples - hrs;
          const tb = singles + doubles * 2 + triples * 3 + hrs * 4;
          const obpDenom = ab + bb + hbp + sf;
          result.obp = obpDenom > 0 ? Math.round((h + bb + hbp) / obpDenom * 1000) / 1000 : 0;
          result.slg = ab > 0 ? Math.round(tb / ab * 1000) / 1000 : 0;
          result.pas = parseInt(bs.plateAppearances) || result.pas;
          break;
        }
      }
    }

    // Enrich with Savant xwOBA and VAA
    const savRows = isGame ? savantData : seasonSavant;
    if (savRows && savRows.length) {
      const type = isPitcher ? "pitcher" : "batter";
      const { xwoba, pitchVAAs } = enrichWithSavant(savRows, selectedPlayer.id, type);

      // Skip Savant xwOBA for AAA (returns wrong MLB-level data); use prospect CSV instead
      if (xwoba != null && !isAAA) result.xwoba = xwoba;
      if (Object.keys(pitchVAAs).length > 0) result.pitchVAAs = pitchVAAs;

      if (result.pitches && Object.keys(pitchVAAs).length > 0) {
        result.pitches = result.pitches.map(p => ({
          ...p,
          vaa: pitchVAAs[p.pitchType] ?? p.vaa,
        }));
      }
    }

    // Fallback: use bulk Savant xwOBA if per-game enrichment didn't provide it
    if (result.xwoba == null && !isAAA && bulkXwoba && selectedPlayer.id) {
      const bx = bulkXwoba.get(selectedPlayer.id);
      if (bx != null) result.xwoba = Math.round(bx * 1000) / 1000;
    }

    // AAA: use prospect xwOBA CSV data as primary source
    if (result.xwoba == null && prospectXwoba && selectedPlayer?.name) {
      const playerName = selectedPlayer.name;
      const yr = String(currentSeason);
      let entry = prospectXwoba[playerName];
      if (!entry) {
        const normName = norm(playerName);
        for (const [k, v] of Object.entries(prospectXwoba)) {
          if (norm(k) === normName) { entry = v; break; }
        }
      }
      if (entry && entry[yr] != null) result.xwoba = entry[yr];
    }

    // ── Apply pitch reclassifications (session overrides) ──
    // pitchOverrides take precedence over typeOverrides so a single-pitch
    // change overrules a bulk change. Reclassified pitches are flagged with
    // _reclassified so the movement plot can render them with a thicker stroke.
    if (isPitcher && result.pitches?.length && (pitchOverrides.size || typeOverrides.size)) {
      result.pitches = result.pitches.map(p => {
        const key = pitchOverrideKey(p);
        if (pitchOverrides.has(key)) {
          const newType = pitchOverrides.get(key);
          return newType !== p.pitchType
            ? { ...p, pitchType: newType, _reclassified: true }
            : p;
        }
        if (typeOverrides.has(p.pitchType)) {
          const newType = typeOverrides.get(p.pitchType);
          return newType !== p.pitchType
            ? { ...p, pitchType: newType, _reclassified: true }
            : p;
        }
        return p;
      });
    }

    return result;
  }, [extractedData, savantData, seasonSavant, selectedPlayer, isGame, isPitcher, isAAA, boxscoreData, bulkXwoba, prospectXwoba, currentSeason, pitchOverrides, typeOverrides]);

  const hasData = enrichedData && (isPitcher ? enrichedData.totalPitches > 0 : enrichedData.pas > 0);

  // Live Pitch+ scoring: send loaded pitches to /score_aggregate, which normalizes
  // at the pitcher×pitch-type level (same scale as /leaderboard pre-computed values).
  // Three parallel requests: all pitches (pitch table), vs-LHB only, vs-RHB only
  // (the latter two feed PlatoonUsageBars per-handedness Pitch+).
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!isPitcher || !enrichedData?.pitches?.length || !selectedPlayer) {
        setPitchPlus(null);
        return;
      }

      let pitcherHand = selectedPlayer.pitchHand?.code;
      if (pitcherHand !== "L" && pitcherHand !== "R") {
        try {
          const r = await fetch(`/mlb-api/api/v1/people/${selectedPlayer.id}`);
          const j = await r.json();
          pitcherHand = j?.people?.[0]?.pitchHand?.code || "R";
        } catch { pitcherHand = "R"; }
      }

      const makePitch = (p) => ({
        pitcher_id: selectedPlayer.id,
        _stand: p.batSide || "R",
        _p_throws: pitcherHand,
        _pitchType: p.pitchType,
        details: { type: { code: scorePitchCode(p.pitchType) } },
        pitchData: {
          startSpeed: p.velo,
          extension: p.extension,
          strikeZoneTop: p.szTop,
          strikeZoneBottom: p.szBot,
          coordinates: {
            pfxX: p.pfxX_raw ?? null,
            pfxZ: p.pfxZ_raw ?? null,
            pX: p.pX, pZ: p.pZ,
            x0: p.relX, z0: p.relHeight,
            vX0: p.vX0, vY0: p.vY0, vZ0: p.vZ0,
            aX: p.aX, aY: p.aY, aZ: p.aZ,
          },
          breaks: { spinRate: p.spin, spinDirection: p.spinDirection },
        },
      });

      const eligible = enrichedData.pitches.filter(
        p => p.velo != null && p.pX != null && p.pZ != null && p.pitchType !== "UN"
      );
      if (!eligible.length) { setPitchPlus(null); return; }

      const allPitches = eligible.map(makePitch);
      const lPitches   = eligible.filter(p => p.batSide === "L").map(makePitch);
      const rPitches   = eligible.filter(p => p.batSide === "R").map(makePitch);

      // Map API-aliased codes (FO→FS, CS→CU) back to original pitch codes for output keys
      const aliasedToOrig = {};
      for (const p of allPitches) {
        const apiCode = p.details.type.code;
        if (!aliasedToOrig[apiCode]) aliasedToOrig[apiCode] = p._pitchType;
      }

      const post = (pitches) => {
        if (!pitches.length) return Promise.resolve(null);
        return fetch("https://pitch-plus-api.onrender.com/score_aggregate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pitches }),
        }).then(r => r.ok ? r.json() : null).catch(() => null);
      };

      const [allData, lData, rData] = await Promise.all([
        post(allPitches),
        post(lPitches),
        post(rPitches),
      ]);

      if (cancelled) return;

      const byType  = allData?.by_pitch_type || {};
      const byTypeL = lData?.by_pitch_type   || {};
      const byTypeR = rData?.by_pitch_type   || {};

      const round = (v) => v != null ? Math.round(v * 10) / 10 : null;
      const out = {};
      for (const [apiPt, v] of Object.entries(byType)) {
        const origPt = aliasedToOrig[apiPt] || apiPt;
        out[origPt] = {
          stuffPlus:  round(v.stuff),
          locPlus:    round(v.loc),
          tunnelPlus: round(v.tun),
          pitchPlus:  round(v.pitch),
          L: { pitchPlus: round(byTypeL[apiPt]?.pitch) },
          R: { pitchPlus: round(byTypeR[apiPt]?.pitch) },
        };
      }
      console.log(`[Pitch+] Scored ${allPitches.length} pitches via /score_aggregate:`, out);
      setPitchPlus(out);
    })();
    return () => { cancelled = true; };
  }, [enrichedData, selectedPlayer, isPitcher, isAAA]);

  // For Regular Season MLB season view, override per-type Plus values with
  // pre-computed season aggregates from /leaderboard. These cover the full
  // season corpus and are the authoritative source for RS pitcher grades.
  // L/R per-handedness Pitch+ (platoon splits) are preserved from /score_aggregate.
  useEffect(() => {
    if (seasonType !== "R" || isAAA || isGame || !selectedPlayer?.id) {
      setLeaderboardPlus(null);
      return;
    }
    let cancelled = false;
    fetch(`https://pitch-plus-api.onrender.com/leaderboard?season=${currentSeason}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (cancelled || !data?.pitchers) return;
        const entry = data.pitchers.find(p => p.player_id === Number(selectedPlayer.id));
        if (!entry?.by_pitch_type) { setLeaderboardPlus(null); return; }
        setLeaderboardPlus(entry.by_pitch_type);
      })
      .catch(() => setLeaderboardPlus(null));
    return () => { cancelled = true; };
  }, [seasonType, isAAA, isGame, selectedPlayer?.id, currentSeason]);

  // Merge pre-computed Plus values over /score_aggregate values.
  // Leaderboard wins on stuffPlus/locPlus/tunnelPlus/pitchPlus (full season corpus).
  // /score_aggregate wins on L/R splits (not in season aggregates).
  const effectivePitchPlus = useMemo(() => {
    if (!pitchPlus) return pitchPlus;
    if (!leaderboardPlus) return pitchPlus;
    const merged = { ...pitchPlus };
    for (const [pt, pg] of Object.entries(leaderboardPlus)) {
      merged[pt] = {
        ...(merged[pt] || {}),
        stuffPlus:  pg.stuff  ?? merged[pt]?.stuffPlus,
        locPlus:    pg.loc    ?? merged[pt]?.locPlus,
        tunnelPlus: pg.tun    ?? merged[pt]?.tunnelPlus,
        pitchPlus:  pg.pitch  ?? merged[pt]?.pitchPlus,
      };
    }
    return merged;
  }, [pitchPlus, leaderboardPlus]);

  return (
    <div style={{ padding: "16px 20px" }}>
      <div style={{ display: "flex", gap: 2, marginBottom: 12, flexWrap: "wrap", alignItems: "center" }}>
        {SEASON_TYPES.map(st => (
          <button key={st.id} onClick={() => { setSeasonType(st.id); if (st.id !== "R") setLevel("MLB"); }} style={{
            padding: "4px 10px", fontSize: 10, fontWeight: seasonType === st.id ? 700 : 400,
            background: seasonType === st.id ? "#f59e0b" : "transparent",
            color: seasonType === st.id ? "#fff" : "#666",
            border: seasonType === st.id ? "none" : "1px solid #333",
            borderRadius: 4, cursor: "pointer", fontFamily: "inherit",
          }}>{st.label}</button>
        ))}
        {seasonType === "R" && (
          <>
            <div style={{ width: 1, height: 22, background: t.divider, margin: "0 4px" }} />
            {["MLB", "AAA"].map(lv => (
              <button key={lv} onClick={() => setLevel(lv)} style={{
                padding: "4px 10px", fontSize: 10, fontWeight: level === lv ? 700 : 400,
                background: level === lv ? "#2563eb" : "transparent",
                color: level === lv ? "#fff" : "#666",
                border: level === lv ? "none" : "1px solid #333",
                borderRadius: 4, cursor: "pointer", fontFamily: "inherit",
              }}>{lv}</button>
            ))}
          </>
        )}
      </div>

      {error && <div style={{ color: "#f59e0b", fontSize: 12, marginBottom: 8 }}>{error}</div>}

      {isGame && (
        <div>
          <div style={{ display: "flex", gap: 12, marginBottom: 12, flexWrap: "wrap", alignItems: "center" }}>
            <select value={selectedGame?.gamePk || ""} onChange={e => { const g = games.find(x => x.gamePk === parseInt(e.target.value)); setSelectedGame(g || null); setSelectedPlayer(null); }} style={{ padding: "6px 12px", background: t.inputBg, color: t.textSecondary, border: `1px solid ${t.inputBorder}`, borderRadius: 6, fontSize: 12, fontFamily: "inherit", minWidth: 280 }}>
              <option value="">Select Game...</option>
              {games.map(g => {
                const awayName = g.awayFlag ? `${g.awayFlag} ${g.away?.teamName || g.away?.abbreviation || "?"}` : (g.away?.abbreviation || g.away?.teamName || "?");
                const homeName = g.homeFlag ? `${g.homeFlag} ${g.home?.teamName || g.home?.abbreviation || "?"}` : (g.home?.abbreviation || g.home?.teamName || "?");
                const prefix = g.isLive ? "🔴 LIVE — " : g.isExhibition ? "⚾ " : "";
                const score = g.awayScore != null ? ` (${g.awayScore}-${g.homeScore})` : "";
                return <option key={g.gamePk} value={g.gamePk}>{prefix}{g.date} — {awayName} @ {homeName}{score}</option>;
              })}
            </select>
            {loadingGames && <span style={{ fontSize: 11, color: t.textMuted }}>Loading schedule...</span>}
            {loadingPbp && <span style={{ fontSize: 11, color: t.textMuted }}>Loading game data...</span>}
          </div>
          {selectedGame && boxscoreData && (
            <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
              <SearchableSelect
                value={selectedPlayer?.id || ""}
                options={gamePlayers.map(p => ({
                  value: p.id,
                  label: `${p.name} (${p.team})${isPitcher ? ` — ${p.ip} IP` : ` — ${p.hits}/${p.abs}`}`,
                }))}
                onChange={(_, opt) => {
                  const p = gamePlayers.find(x => x.id === opt?.value);
                  setSelectedPlayer(p || null);
                }}
                placeholder={`Search ${isPitcher ? "pitcher" : "hitter"}…`}
                style={{ padding: "6px 12px", background: t.inputBg, color: t.textSecondary, border: `1px solid ${t.inputBorder}`, borderRadius: 6, fontSize: 12, minWidth: 280 }}
              />
            </div>
          )}
          {!selectedGame && !loadingGames && <div style={{ color: t.textMuted, textAlign: "center", padding: 40 }}>Select a game</div>}
          {selectedGame && !boxscoreData && loadingPbp && <div style={{ color: t.textMuted, textAlign: "center", padding: 40 }}>Loading game data...</div>}
          {selectedGame && boxscoreData && !selectedPlayer && <div style={{ color: t.textMuted, textAlign: "center", padding: 40 }}>Select a {isPitcher ? "pitcher" : "hitter"} from the game</div>}
        </div>
      )}

      {!isGame && (
        <div>
          <div style={{ display: "flex", gap: 12, marginBottom: 12, alignItems: "center" }}>
            <SearchableSelect
              value={selectedPlayer?.id || ""}
              options={playerList.map(p => ({ value: p.id, label: `${p.name} (${p.team})` }))}
              onChange={(_, opt) => {
                const p = playerList.find(x => x.id === opt?.value);
                setSelectedPlayer(p || null);
              }}
              placeholder={`Search ${isPitcher ? "pitcher" : "hitter"}…`}
              style={{ padding: "6px 12px", background: t.inputBg, color: t.textSecondary, border: `1px solid ${t.inputBorder}`, borderRadius: 6, fontSize: 12, minWidth: 280 }}
            />
            {loadingPlayers && <span style={{ fontSize: 11, color: t.textMuted }}>Loading players...</span>}
            {loadingSeason && <span style={{ fontSize: 11, color: t.textMuted }}>{seasonProgress}</span>}
          </div>
          {!selectedPlayer && <div style={{ color: t.textMuted, textAlign: "center", padding: 40 }}>Select a {isPitcher ? "pitcher" : "hitter"}</div>}
        </div>
      )}

      {enrichedData && !hasData && <div style={{ color: t.textMuted, textAlign: "center", padding: 40 }}>{selectedPlayer?.name} did not {isPitcher ? "pitch" : "bat"} in this {isGame ? "game" : "period"}</div>}

      {/* Reclassify toggle bar — only visible for pitcher views with data.
          Lives outside cardRef so the saved PNG doesn't include it. */}
      {hasData && isPitcher && !isCounts && (
        <div style={{
          maxWidth: 1000, margin: "0 auto 8px",
          display: "flex", justifyContent: "flex-end", alignItems: "center", gap: 12,
          fontSize: 11,
        }}>
          {(pitchOverrides.size > 0 || typeOverrides.size > 0) && (
            <button
              onClick={() => { setPitchOverrides(new Map()); setTypeOverrides(new Map()); }}
              style={{
                padding: "5px 10px", fontSize: 11, fontWeight: 600,
                background: "transparent", color: t.textMuted,
                border: `1px solid ${t.divider}`, borderRadius: 5, cursor: "pointer",
              }}
              title="Restore original pitch types"
            >
              Clear ({pitchOverrides.size + typeOverrides.size})
            </button>
          )}
          <button
            onClick={() => setReclassifyMode(m => !m)}
            style={{
              padding: "6px 14px", fontSize: 11, fontWeight: 700,
              background: reclassifyMode ? t.text : t.inputBg,
              color: reclassifyMode ? t.cardBg : t.textSecondary,
              border: `1px solid ${reclassifyMode ? t.text : t.inputBorder}`,
              borderRadius: 5, cursor: "pointer",
              letterSpacing: "0.04em", textTransform: "uppercase",
            }}
            title="Click pitches in the movement plot to reclassify"
          >
            {reclassifyMode ? "✓ Reclassify on" : "Reclassify"}
          </button>
        </div>
      )}

      {hasData && isPitcher && !isCounts && (
        <PitcherView
          data={enrichedData} player={selectedPlayer}
          game={isGame ? selectedGame : null}
          season={currentSeason} seasonType={seasonType}
          isGame={isGame} isAAA={isAAA}
          leagueAvgs={effectiveLeagueAvgs} pitchPlus={effectivePitchPlus}
          reclassifyMode={reclassifyMode}
          onPitchClick={(p) => setReclassifyTarget({ kind: "single", pitch: p })}
          onTypeClick={(type, count) => setReclassifyTarget({ kind: "bulk", type, count })}
        />
      )}
      {hasData && !isPitcher && !isCounts && <HitterView data={enrichedData} player={selectedPlayer} game={isGame ? selectedGame : null} season={currentSeason} seasonType={seasonType} isGame={isGame} isAAA={isAAA} leagueAvgs={effectiveLeagueAvgs} />}
      {hasData && isPitcher && isCounts && (
        <CountsCard player={selectedPlayer} season={currentSeason} seasonType={seasonType} isAAA={isAAA} isPitcher={true}>
          <PitcherCountTool pitches={enrichedData.pitches} leagueAvgs={effectiveLeagueAvgs} isAAA={isAAA} />
        </CountsCard>
      )}
      {hasData && !isPitcher && isCounts && (
        <CountsCard player={selectedPlayer} season={currentSeason} seasonType={seasonType} isAAA={isAAA} isPitcher={false}>
          <HitterCountTool pitches={enrichedData.pitches} leagueAvgs={effectiveLeagueAvgs} isAAA={isAAA} />
        </CountsCard>
      )}

      {/* Reclassify modal — global so it overlays everything. The target
          object's `kind` field decides whether we update pitchOverrides
          (single) or typeOverrides (bulk reclassification). */}
      {reclassifyTarget && (
        <ReclassifyModal
          target={reclassifyTarget}
          onClose={() => setReclassifyTarget(null)}
          onPick={(newType) => {
            if (reclassifyTarget.kind === "bulk") {
              const origType = reclassifyTarget.type;
              setTypeOverrides(prev => {
                const next = new Map(prev);
                if (newType === origType) {
                  // Picking the same type clears any existing override.
                  next.delete(origType);
                } else {
                  next.set(origType, newType);
                }
                return next;
              });
              // Bulk reclassification supersedes single overrides for those
              // pitches: drop any single overrides whose original type matches.
              setPitchOverrides(prev => {
                if (!prev.size) return prev;
                // We don't have original-type info on the override key, so we
                // just leave them alone — the apply step prefers single over
                // bulk anyway, which is the right precedence.
                return prev;
              });
            } else {
              const pitch = reclassifyTarget.pitch;
              setPitchOverrides(prev => {
                const next = new Map(prev);
                next.set(pitchOverrideKey(pitch), newType);
                return next;
              });
            }
            setReclassifyTarget(null);
          }}
        />
      )}
    </div>
  );
}

function CountsCard({ player, season, seasonType, isAAA, isPitcher, children }) {
  const { theme: t } = useTheme();
  const cardRef = useRef(null);
  const subtitle = `${season} ${GAME_TYPE_LABELS[seasonType]} Counts`;
  const saveCard = async () => {
    await saveCardAsPng(cardRef, `${player.name.replace(/\s+/g, "_")}_${isPitcher ? "pitching" : "hitting"}_counts.png`);
  };
  return (
    <div>
      <div ref={cardRef} style={{ background: t.cardBg, borderRadius: 12, border: `1px solid ${t.cardBorder}`, overflow: "hidden", width: 900, margin: "0 auto", boxShadow: `0 4px 24px ${t.shadow}` }}>
        <SummaryHeader player={player} subtitle={subtitle} seasonType={seasonType} isAAA={isAAA} />
        {children}
        <div style={{ padding: "8px 20px 10px", display: "flex", justifyContent: "space-between", fontSize: 10, color: t.textFaint }}>
          <span>Created by: @PastTheEyeTest on X</span>
          <span style={{ fontStyle: "italic" }}>Data: Baseball Savant / MLB Stats API{isAAA ? " / Prospect Savant" : ""}</span>
        </div>
      </div>
      <div style={{ textAlign: "center", marginTop: 12 }}>
        <button onClick={saveCard} style={{ padding: "6px 16px", fontSize: 11, fontWeight: 600, background: t.inputBg, color: t.textMuted, border: `1px solid ${t.inputBorder}`, borderRadius: 6, cursor: "pointer" }}>📥 Save as PNG</button>
      </div>
    </div>
  );
}

function PitcherView({ data, player, game, season, seasonType, isGame, isAAA, leagueAvgs, pitchPlus, reclassifyMode = false, onPitchClick = null, onTypeClick = null }) {
  const { theme: t } = useTheme();
  const cardRef = useRef(null);
  const pitchRows = useMemo(() => aggregateByPitchType(data.pitches), [data.pitches]);
  // Per-side panel selection. Default = current "zones" behavior on both sides.
  const [leftPanel, setLeftPanel] = useState("zones");   // "zones" | "velo"
  const [rightPanel, setRightPanel] = useState("zones"); // "zones" | "platoon"
  const strikePct = data.totalPitches > 0 ? Math.round(data.pitches.filter(p => p.isStrike || p.isInPlay).length / data.totalPitches * 1000) / 10 : 0;
  const swings = data.pitches.filter(p => p.isSwing);
  const whiffPct = swings.length > 0 ? Math.round(data.pitches.filter(p => p.isWhiff).length / swings.length * 1000) / 10 : 0;
  const subtitle = (() => {
    if (isGame && game) {
      const home = game.home?.abbreviation || game.home?.teamName || "?";
      const away = game.away?.abbreviation || game.away?.teamName || "?";
      const [y, m, d] = (game.date || "").split("-");
      const dateStr = y && m && d ? `${m}/${d}/${y}` : (game.date || "");
      return `${home} vs. ${away} — ${dateStr}`;
    }
    return `${season} ${GAME_TYPE_LABELS[seasonType]}`;
  })();
  const stats = isGame ? [
    { label: "IP", value: data.ip, format: ".1f" },
    { label: "H", value: data.hits }, { label: "R", value: data.runs },
    { label: "ER", value: data.ers }, { label: "K", value: data.ks },
    { label: "BB/HBP", value: (data.bbs || 0) + (data.hbps || 0) },
    { label: "Strike%", value: strikePct, format: ".1f" },
    { label: "Whiff%", value: whiffPct, format: ".1f" },
  ] : (() => {
    // Season: show ERA + FIP instead of H/R
    // FIP = (13×HR + 3×(BB+HBP) - 2×K) / IP + constant
    const FIP_CONSTANTS = {
      2023: 3.107, 2024: 3.102, 2025: 3.150, 2026: 3.150,
    };
    const cFIP = FIP_CONSTANTS[season] || FIP_CONSTANTS[2025] || 3.15;
    const ipActual = Math.floor(data.ip) + (((data.ip % 1) * 10) / 3);
    const bbHbp = (data.bbs || 0) + (data.hbps || 0);
    const fip = ipActual > 0 ? ((13 * (data.hrs || 0) + 3 * bbHbp - 2 * data.ks) / ipActual) + cFIP : null;
    const era = ipActual > 0 ? (data.ers * 9) / ipActual : null;
    const bf = data.bf || 0;
    return [
      { label: "IP", value: data.ip, format: ".1f" },
      { label: "ERA", value: era, format: ".2f", good: "low", thresholds: isAAA ? [3.6, 4.5, 5.5] : [3.0, 3.7, 4.4] },
      { label: "FIP", value: fip, format: ".2f", good: "low", thresholds: isAAA ? [3.5, 4.3, 5.2] : [3.0, 3.7, 4.4] },
      { label: "K%", value: bf > 0 ? data.ks / bf * 100 : null, format: ".1f", good: "high", thresholds: isAAA ? [17, 21, 26] : [20, 24, 28] },
      { label: "BB%", value: bf > 0 ? data.bbs / bf * 100 : null, format: ".1f", good: "low", thresholds: isAAA ? [6, 9, 12] : [5, 7.5, 10] },
      { label: "Strike%", value: strikePct, format: ".1f" },
      { label: "Whiff%", value: whiffPct, format: ".1f" },
    ];
  })();
  const saveCard = async () => {
    await saveCardAsPng(cardRef, `${player.name.replace(/\s+/g, "_")}_pitching_${isGame ? "game" : "season"}.png`);
  };
  return (
    <div>
      {/* Panel selectors live OUTSIDE cardRef so they don't appear in the saved PNG */}
      <div style={{
        maxWidth: 1000, margin: "0 auto 10px", display: "flex",
        justifyContent: "space-between", gap: 12, flexWrap: "wrap",
      }}>
        <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: t.textMuted, fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase" }}>
          Left panel
          <select
            value={leftPanel}
            onChange={e => setLeftPanel(e.target.value)}
            style={{
              padding: "5px 10px", background: t.inputBg, color: t.textSecondary,
              border: `1px solid ${t.inputBorder}`, borderRadius: 6, fontSize: 12,
              fontFamily: "inherit", fontWeight: 600, outline: "none", minWidth: 140,
            }}
          >
            <option value="zones">Zones — vs LHB</option>
            <option value="zonesAll">Zones — overall</option>
            <option value="velo">Velocity — by pitch #{!isGame ? " / outing" : ""}</option>
          </select>
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: t.textMuted, fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase" }}>
          Right panel
          <select
            value={rightPanel}
            onChange={e => setRightPanel(e.target.value)}
            style={{
              padding: "5px 10px", background: t.inputBg, color: t.textSecondary,
              border: `1px solid ${t.inputBorder}`, borderRadius: 6, fontSize: 12,
              fontFamily: "inherit", fontWeight: 600, outline: "none", minWidth: 140,
            }}
          >
            <option value="zones">Zones — vs RHB</option>
            <option value="zonesAll">Zones — overall</option>
            <option value="platoon">Usage — by platoon</option>
          </select>
        </label>
      </div>

      <div ref={cardRef} style={{ background: t.cardBg, borderRadius: 12, border: `1px solid ${t.cardBorder}`, overflow: "hidden", maxWidth: 1000, margin: "0 auto", boxShadow: `0 4px 24px ${t.shadow}` }}>
        <SummaryHeader player={player} subtitle={subtitle} seasonType={seasonType} isAAA={isAAA} />
        <StatBar stats={stats} />
        <div style={{ display: "flex", justifyContent: "center", alignItems: "center", padding: "8px 4px 0", gap: 4 }}>
          <div style={{ width: 260 }}>
            {leftPanel === "zones" ? (
              <LocationZonePanel pitches={data.pitches.filter(p => p.batSide === "L")} side="L" width={260} isGame={isGame} />
            ) : leftPanel === "zonesAll" ? (
              <LocationZonePanel pitches={data.pitches} side="ALL" width={260} isGame={isGame} />
            ) : (
              <RollingVeloChart pitches={data.pitches} mode={isGame ? "game" : "season"} width={260} height={400} />
            )}
          </div>
          <MovementPlot
            pitches={data.pitches}
            width={420} height={400} maxPitches={200}
            onPitchClick={reclassifyMode ? onPitchClick : null}
          />
          <div style={{ width: 260 }}>
            {rightPanel === "zones" ? (
              <LocationZonePanel pitches={data.pitches.filter(p => p.batSide === "R")} side="R" width={260} isGame={isGame} />
            ) : rightPanel === "zonesAll" ? (
              <LocationZonePanel pitches={data.pitches} side="ALL" width={260} isGame={isGame} />
            ) : (
              <PlatoonUsageBars pitches={data.pitches} pitchPlus={pitchPlus} width={260} height={400} />
            )}
          </div>
        </div>
        <PitchTypeLegend types={[...new Set(data.pitches.map(p => p.pitchType))].filter(t => t !== "UN")} />
        <PitchTable
          rows={pitchRows} leagueAvgs={leagueAvgs} isAAA={isAAA} pitchPlus={pitchPlus}
          onTypeClick={reclassifyMode ? onTypeClick : null}
        />
        <div style={{ padding: "6px 16px 8px", display: "flex", justifyContent: "space-between", fontSize: 10, color: t.textFaint }}>
          <span>Created by: @PastTheEyeTest on X</span>
          <span style={{ fontStyle: "italic" }}>Data: Baseball Savant / MLB Stats API{isAAA ? " / Prospect Savant" : ""}</span>
        </div>
      </div>
      <div style={{ textAlign: "center", marginTop: 12 }}>
        <button onClick={saveCard} style={{ padding: "6px 16px", fontSize: 11, fontWeight: 600, background: t.inputBg, color: t.textMuted, border: `1px solid ${t.inputBorder}`, borderRadius: 6, cursor: "pointer" }}>📥 Save as PNG</button>
      </div>
    </div>
  );
}

function HitterView({ data, player, game, season, seasonType, isGame, isAAA, leagueAvgs }) {
  const { theme: t } = useTheme();
  const cardRef = useRef(null);
  const subtitle = (() => {
    if (isGame && game) {
      const home = game.home?.abbreviation || game.home?.teamName || "?";
      const away = game.away?.abbreviation || game.away?.teamName || "?";
      const [y, m, d] = (game.date || "").split("-");
      const dateStr = y && m && d ? `${m}/${d}/${y}` : (game.date || "");
      return `${home} vs. ${away} — ${dateStr}`;
    }
    return `${season} ${GAME_TYPE_LABELS[seasonType]}`;
  })();

  const stats = [
    // xwOBA only for RS/PS — Savant doesn't compute expected stats for ST/WBC
    // AAA thresholds calibrated from Prospect Savant percentiles (25th/50th/75th)
    ...((seasonType === "R" || seasonType === "P") ? [{ label: "xwOBA", value: data.xwoba, format: ".3f", good: "high", thresholds: isAAA ? [0.275, 0.305, 0.345] : [0.295, 0.315, 0.355] }] : []),
    { label: "OBP", value: data.obp, format: ".3f", good: "high", thresholds: isAAA ? [0.310, 0.340, 0.380] : [0.290, 0.320, 0.370] },
    { label: "SLG", value: data.slg, format: ".3f", good: "high", thresholds: isAAA ? [0.380, 0.430, 0.500] : [0.370, 0.430, 0.500] },
    { label: "Avg EV", value: data.avgEV, format: ".1f", good: "high", thresholds: isAAA ? [84, 87.5, 91] : [86, 89, 92] },
    { label: "Max EV", value: data.maxEV, format: ".1f", good: "high", thresholds: isAAA ? [100, 107, 114] : [102, 107, 112] },
    { label: "K%", value: data.kPct, format: ".1f", good: "low", thresholds: isAAA ? [19, 23, 28] : [16, 21, 26] },
    { label: "BB%", value: data.bbPct, format: ".1f", good: "high", thresholds: isAAA ? [7, 10, 14] : [5, 8.5, 12] },
    { label: "Chase%", value: data.chasePct, format: ".1f", good: "low", thresholds: isAAA ? [23, 28, 33] : [23, 28, 33] },
    { label: "Whiff%", value: data.whiffPct, format: ".1f", good: "low", thresholds: isAAA ? [22, 28, 34] : [19, 25, 31] },
  ];
  const battedBalls = data.pitches.filter(p => p.isInPlay && p.hitData);
  const hardContact = data.pitches.filter(p => p.isInPlay && p.hitData?.launchSpeed >= 95);
  const whiffs = data.pitches.filter(p => p.isWhiff);
  const pitchTypes = [...new Set(data.pitches.map(p => p.pitchType))];
  const saveCard = async () => {
    await saveCardAsPng(cardRef, `${player.name.replace(/\s+/g, "_")}_hitting_${isGame ? "game" : "season"}.png`);
  };
  return (
    <div>
      <div ref={cardRef} style={{ background: t.cardBg, borderRadius: 12, border: `1px solid ${t.cardBorder}`, overflow: "hidden", maxWidth: 700, margin: "0 auto", boxShadow: `0 4px 24px ${t.shadow}` }}>
        <SummaryHeader player={player} subtitle={subtitle} seasonType={seasonType} isAAA={isAAA} />
        <StatBar stats={stats} />
        <div style={{ display: "flex", justifyContent: "center" }}>
          <SprayChart battedBalls={battedBalls} width={660} height={380} />
        </div>
        <div style={{ display: "flex", justifyContent: "center", gap: 12, padding: "8px 16px" }}>
          <ZonePlot pitches={hardContact} title="95+ EV Contact" width={310} height={242} />
          <ZonePlot pitches={whiffs} title="Whiffs" width={310} height={242} />
        </div>
        <PitchTypeLegend types={pitchTypes} />
        <div style={{ padding: "6px 16px 8px", display: "flex", justifyContent: "space-between", fontSize: 10, color: t.textFaint }}>
          <span>Created by: @PastTheEyeTest on X</span>
          <span style={{ fontStyle: "italic" }}>Data: Baseball Savant / MLB Stats API{isAAA ? " / Prospect Savant" : ""}</span>
        </div>
      </div>
      <div style={{ textAlign: "center", marginTop: 12 }}>
        <button onClick={saveCard} style={{ padding: "6px 16px", fontSize: 11, fontWeight: 600, background: t.inputBg, color: t.textMuted, border: `1px solid ${t.inputBorder}`, borderRadius: 6, cursor: "pointer" }}>📥 Save as PNG</button>
      </div>
    </div>
  );
}

function SummaryHeader({ player, subtitle, seasonType, isAAA }) {
  const headshot = player.id ? `/mlb-photos/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,h_213,c_thumb,g_face,q_auto:best/v1/people/${player.id}/headshot/67/current` : null;
  const logo = getLogoUrl(player.team, player.teamId);
  const flag = (!isAAA && seasonType === "W") ? getWbcFlag(player.team) : null;
  const showFlag = seasonType === "W" ? (flag || null) : null;
  const showLogo = seasonType === "W" ? (!flag ? logo : null) : logo;

  const bgColor = MLB_TEAM_PRIMARY[player.team] || "#1e293b";
  const light = hexLuminance(bgColor) > 0.179;
  const textColor = light ? "#111111" : "#ffffff";
  const subColor = light ? "rgba(0,0,0,0.6)" : "rgba(255,255,255,0.72)";
  const ringColor = light ? "rgba(0,0,0,0.2)" : "rgba(255,255,255,0.25)";
  const initColor = light ? "rgba(0,0,0,0.25)" : "rgba(255,255,255,0.2)";

  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "16px 24px 14px", gap: 20, background: bgColor }}>
      <div style={{ width: 72, height: 72, borderRadius: "50%", background: ringColor, overflow: "hidden", flexShrink: 0, border: `2px solid ${ringColor}`, display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
        <span style={{ fontSize: 22, fontWeight: 700, color: initColor, position: "absolute" }}>{player.name ? player.name.split(" ").map(w => w[0]).join("").toUpperCase().slice(0, 2) : ""}</span>
        {headshot && <img src={headshot} alt="" style={{ width: "100%", height: "100%", objectFit: "cover", position: "relative", zIndex: 1 }} onError={e => { e.target.style.display = "none"; }} />}
      </div>
      <div style={{ flex: 1, textAlign: "center" }}>
        <div style={{ fontSize: 24, fontWeight: 800, color: textColor, letterSpacing: "-0.02em" }}>{player.name}</div>
        <div style={{ fontSize: 12, color: subColor, marginTop: 4, letterSpacing: "0.06em", fontStyle: "italic", fontWeight: 600, textTransform: "uppercase" }}>{subtitle}</div>
      </div>
      <div style={{ width: 80, height: 80, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
        {showLogo && <img src={showLogo} alt={player.team} style={{ width: 80, height: 80, objectFit: "contain", filter: "drop-shadow(0 2px 8px rgba(0,0,0,0.4))" }} onError={e => { e.target.style.display = "none"; }} />}
        {!showLogo && showFlag && <span style={{ fontSize: 48, lineHeight: 1 }}>{showFlag}</span>}
      </div>
    </div>
  );
}
